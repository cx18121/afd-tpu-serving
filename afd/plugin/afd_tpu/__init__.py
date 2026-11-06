"""Attention-FFN disaggregation (AFD) for vLLM + vllm-torchtpu, as a vLLM
general plugin. Enabled with AFD_ROLE_SPLIT=1; does nothing otherwise.

Layout: data parallel 2 x tensor parallel 4 with expert parallelism, 8 chips.
  attention replica (DP rank AFD_ATTN_DP_RANK, default 0): serves every
      request (clients send header X-data-parallel-rank=0); its chips hold no
      active experts, so in each MoE layer they compute nothing.
  expert replica (the other DP rank): gets no requests; vLLM keeps it in step
      with dummy batches. Its chip j holds experts [32j, 32j+32) and computes
      them for the attention replica's tokens.

Dispatch and combine are vLLM's existing DP collectives around each MoE
layer: an all-gather of tokens across the two replicas, then a reduce-scatter
that sums each token's partial results back onto its replica. Attention chip
j then holds the contribution of experts [32j, 32j+32), and the tensor-
parallel all-reduce over the attention replica's 4 chips adds all 128.

Why it is built this way (all measured on TPU v6e, see afd/transfer_test.py):
- Different compiled programs per role cannot share a collective on
  torch_tpu (hangs, then wrong data), and torch.cond is not supported. So
  every chip runs the same program, and roles differ only in runtime data.
- The role is the expert-range start (`_experts_start`), already a runtime
  buffer: set to 128 on the attention replica, every routed expert is
  "not local" there, so those chips route no tokens through their experts.

Known costs of this first version:
- Attention chips still load 32 experts each (unused), which costs memory.
- The expert replica runs attention on vLLM's dummy batches, and computes
  experts for its own dummy tokens as well as the attention replica's.
"""
import dataclasses
import logging
import os

logger = logging.getLogger("afd_tpu")
_REGISTERED = False
_ROLE_PATCHED = False


def _enabled() -> bool:
    return os.environ.get("AFD_ROLE_SPLIT") == "1"


def register() -> None:
    """vLLM general-plugin entry point; runs in every vLLM process."""
    global _REGISTERED
    if _REGISTERED or not _enabled():
        return
    _REGISTERED = True
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    original_make = FusedMoEParallelConfig.make

    def make(*args, **kwargs):
        config = original_make(*args, **kwargs)
        if not config.use_ep:
            return config
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        # Expert placement by TP rank: chip j of each replica owns the same
        # 1/TP of the experts. Only the expert replica uses them.
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        _patch_attention_role()  # model construction runs in workers only
        return dataclasses.replace(config, ep_size=tp_size, ep_rank=tp_rank)

    FusedMoEParallelConfig.make = staticmethod(make)
    logger.warning("AFD role split enabled (pid %d)", os.getpid())


def _patch_attention_role() -> None:
    """Give the attention replica an empty expert range at load time."""
    global _ROLE_PATCHED
    if _ROLE_PATCHED:
        return
    _ROLE_PATCHED = True
    import torch
    from vllm_torchtpu.layers.adapter import moe_routing

    attn_dp_rank = int(os.environ.get("AFD_ATTN_DP_RANK", "0"))
    original = moe_routing.register_experts_start_buffer

    def register_experts_start_buffer(layer, *, device):
        original(layer, device=device)
        if getattr(layer, "_experts_start", None) is None:
            return
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_tensor_model_parallel_rank,
        )

        dp_rank = get_dp_group().rank_in_group
        role = "attention" if dp_rank == attn_dp_rank else "expert"
        if role == "attention":
            # One past the last expert: no routed expert is local here, so
            # fused_moe_gmm gives every token weight 0 and id -1 on this chip.
            layer.register_buffer(
                "_experts_start",
                torch.tensor(layer.global_num_experts, dtype=torch.int32, device=device),
                persistent=False,
            )
        if not getattr(register_experts_start_buffer, "logged", False):
            register_experts_start_buffer.logged = True
            cfg = layer.moe_config
            logger.warning(
                "AFD role=%s dp_rank=%d tp_rank=%d ep_size=%d ep_rank=%d "
                "local_experts=%d experts_start=%d",
                role, dp_rank, get_tensor_model_parallel_rank(), cfg.ep_size,
                cfg.moe_parallel_config.ep_rank, layer.w13_weight.shape[0],
                int(layer._experts_start.item()),
            )

    moe_routing.register_experts_start_buffer = register_experts_start_buffer
