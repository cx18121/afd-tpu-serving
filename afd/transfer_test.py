#!/usr/bin/env python3
"""AFD feasibility test: can attention chips and expert chips run different
compiled programs and exchange activations every layer, and what does one
exchange cost?

One "layer" of AFD on N = A + F ranks (A attention ranks, F expert ranks):
  attention rank: dispatch its B token rows to every expert rank, then
                  combine: receive each expert rank's partial output, sum.
  expert rank:    receive B rows from every attention rank, multiply by its
                  own weight (a stand-in for its share of the experts), send
                  each result back.
The expected output on attention rank a is sum over expert ranks e of
h_a @ W_e, checked against a float32 CPU reference.

Designs (each runs in its own torchrun, so a hang in one cannot stall the
others):
  hetero   attention and expert ranks compile DIFFERENT programs that share
           the two all_to_all_single collectives. The question this test is
           for: is that allowed inside a compiled step?
  homo     every rank compiles the SAME program; roles are input masks, and
           attention ranks also run the (zero) expert matmul. Works like
           today's TP/EP code; the fallback if hetero is not allowed.
  uneven   hetero with uneven split sizes, so no zero blocks are sent.
  p2p      eager isend/irecv per layer, outside any compiled graph (how
           pipeline-parallel stages hand off today).

Timing: per-layer cost = (time of 1+R chained layers - time of 1) / R, median
of 5 runs, so launch and host overhead cancel out.

  python3 afd/transfer_test.py --launch --platform tpu --out DIR   # on a TPU VM
  python3 afd/transfer_test.py --launch --platform cpu --out DIR   # logic check
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

H = 2048  # Qwen3-30B-A3B hidden size
SEED = 1234
REPS = 16
DESIGNS = ("homo", "cond", "work_both", "work_branch", "p2p", "uneven", "hetero")  # hetero last: it may hang


# ----------------------------------------------------------------- helpers
def expert_weight(e, torch):
    g = torch.Generator().manual_seed(SEED + e)
    return torch.randn(H, H, generator=g) / H**0.5


def attn_input(a, b, torch):
    g = torch.Generator().manual_seed(SEED * 7 + a * 1000 + b)
    return torch.randn(b, H, generator=g)


def a2a(x, group, out_splits=None, in_splits=None):
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), out_splits, in_splits, group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


# ------------------------------------------------------ one layer per design
def make_layer(design, rank, world, n_attn, b, w, group, torch):
    """Returns layer(h) -> h_next for this rank's role. Attention ranks carry
    their tokens through h; expert ranks carry a dummy of the same shape."""
    attn = rank < n_attn
    n_ffn = world - n_attn

    if design == "hetero_ordered":
        # Separate programs per role, with exchange order forced by data
        # dependencies: each program is scheduled on its own, so two
        # collectives with no dependency between them can be issued in either
        # order, and the roles then pair the wrong exchanges. Buffers that are
        # all zeros are taken from the previous exchange's output, not built
        # fresh, so every exchange depends on the one before it.
        if attn:
            def layer(h):
                zero_rows = torch.zeros_like(h)
                send = torch.cat([zero_rows if r < n_attn else h for r in range(world)])
                recv = a2a(send, group)                 # zeros from everyone
                back = a2a(recv, group)                 # combine; send = what we got
                return back[n_attn * b:].reshape(n_ffn, b, H).sum(0) / n_ffn
        else:
            def layer(h):
                # h is what this rank received in the previous combine (zeros),
                # so this dispatch cannot run before that combine.
                recv = a2a(h.repeat(world, 1), group)
                back = a2a(recv @ w, group)
                return back[:b]
        return layer

    if design in ("hetero", "hetero_unused"):
        if attn:
            def layer(h):
                zero = torch.zeros_like(h)
                send = torch.cat([zero if r < n_attn else h for r in range(world)])
                recv = a2a(send, group)               # dispatch (we receive zeros)
                back = a2a(torch.zeros_like(send), group)  # combine
                out = back[n_attn * b:].reshape(n_ffn, b, H).sum(0) / n_ffn
                if design == "hetero_unused":
                    # Original version: the dispatch result is never used, so
                    # the compiler may drop that collective on this side only.
                    return out
                # Use the dispatch result so the collective cannot be removed
                # as dead code (functional collectives count as pure ops).
                return out + recv[:1].sum() * 0
        else:
            def layer(h):
                recv = a2a(torch.zeros(world * b, H, dtype=h.dtype, device=h.device), group)
                y = recv @ w                           # rows from expert ranks are 0
                back = a2a(y, group)
                return h + back[:1].sum() * 0          # keep a data dependency
        return layer

    if design == "homo":
        is_attn = 1.0 if attn else 0.0
        dst_ffn = torch.tensor([0.0 if r < n_attn else 1.0 for r in range(world)])

        def layer(h, is_attn_t, dst_ffn_t):
            send = (h.unsqueeze(0) * dst_ffn_t.view(world, 1, 1).to(h.dtype)).reshape(world * b, H)
            recv = a2a(send * is_attn_t.to(h.dtype), group)
            y = (recv @ w) * (1 - is_attn_t).to(h.dtype)
            back = a2a(y, group)
            out = (back.reshape(world, b, H) * dst_ffn_t.view(world, 1, 1).to(h.dtype)).sum(0) / n_ffn
            return out * is_attn_t.to(h.dtype) + h * (1 - is_attn_t).to(h.dtype)
        layer.extra = (torch.tensor(is_attn), dst_ffn)
        return layer

    if design == "cond":
        # One program on every rank (like homo), but each rank executes only
        # its own role's compute through a runtime branch (torch.cond ->
        # XLA conditional). The two exchanges stay outside the branch, so all
        # ranks run them identically.
        is_attn = torch.tensor(attn)
        dst_ffn = torch.tensor([0.0 if r < n_attn else 1.0 for r in range(world)])

        def build_send(h, d, w_):
            return (h.unsqueeze(0) * d.view(world, 1, 1).to(h.dtype)).reshape(world * b, H)

        def no_send(h, d, w_):
            return torch.zeros(world * b, H, dtype=h.dtype, device=h.device)

        def no_expert(r, w_):
            return torch.zeros_like(r)

        def expert(r, w_):
            return r @ w_

        def combine(back, h, d):
            return (back.reshape(world, b, H) * d.view(world, 1, 1).to(h.dtype)).sum(0) / n_ffn

        def keep(back, h, d):
            return h.clone()

        def layer(h, is_attn_t, dst_ffn_t, w_):
            send = torch.cond(is_attn_t, build_send, no_send, (h, dst_ffn_t, w_))
            recv = a2a(send, group)
            y = torch.cond(is_attn_t, no_expert, expert, (recv, w_))
            back = a2a(y, group)
            return torch.cond(is_attn_t, combine, keep, (back, h, dst_ffn_t))
        layer.extra = (is_attn, dst_ffn, w)
        return layer

    if design in ("work_both", "work_branch"):
        # Skip check for torch.cond: each role has K matmuls of side work that
        # does not feed the exchange. work_both runs both roles' work on every
        # rank (what a plain shared program does); work_branch runs only the
        # rank's own role through torch.cond. If the branch really skips the
        # other role's work, work_branch is ~K matmuls faster per layer.
        k_work = int(os.environ.get("AFD_WORK", "40"))
        base = make_layer("homo", rank, world, n_attn, b, w, group, torch)
        is_attn = torch.tensor(attn)

        def work(z, w_):
            for _ in range(k_work):
                z = z + (z @ w_) * 1e-30
            return z

        def layer(h, is_attn_t, dst_ffn_t, w_, z0):
            if design == "work_both":
                za, ze = work(z0, w_), work(z0 * 2, w_)
                z = torch.where(is_attn_t, za, ze)
            else:
                z = torch.cond(is_attn_t, lambda z_, w2: work(z_, w2),
                               lambda z_, w2: work(z_ * 2, w2), (z0, w_))
            out = base(h, is_attn_t.to(torch.float32), dst_ffn_t)
            return out + z[:1, :1] * 1e-30
        z0 = torch.randn(b, H) * 0.01
        layer.extra = (is_attn, base.extra[1], w, z0)
        return layer

    if design == "uneven":
        to_ffn = [0 if r < n_attn else b for r in range(world)]
        from_attn = [b if r < n_attn else 0 for r in range(world)]
        none = [0] * world
        if attn:
            def layer(h):
                send = torch.cat([h] * n_ffn)
                a2a(send, group, out_splits=none, in_splits=to_ffn)
                back = a2a(torch.zeros(0, H, dtype=h.dtype, device=h.device), group,
                           out_splits=to_ffn, in_splits=none)
                return back.reshape(n_ffn, b, H).sum(0) / n_ffn
        else:
            def layer(h):
                recv = a2a(torch.zeros(0, H, dtype=h.dtype, device=h.device), group,
                           out_splits=from_attn, in_splits=none)
                back = a2a(recv @ w, group, out_splits=none, in_splits=from_attn)
                return h + back.sum() * 0
        return layer

    raise ValueError(design)


def p2p_layer(rank, world, n_attn, b, w, torch):
    """Eager isend/irecv: attention -> every expert rank -> back."""
    import torch.distributed as dist

    attn = rank < n_attn
    n_ffn = world - n_attn
    # Every transfer gets its own tag, as vllm-torchtpu's pipeline ring does
    # (pp_ring._tag): reusing one tag for back-to-back transfers hung on TPU.
    # All ranks call layer() the same number of times, so counters agree.
    count = [0]

    def tags():
        count[0] += 1
        return 2 * count[0], 2 * count[0] + 1  # dispatch tag, combine tag

    def layer(h):
        t_dispatch, t_combine = tags()
        if attn:
            reqs = [dist.isend(h.contiguous(), dst=e, tag=t_dispatch) for e in range(n_attn, world)]
            outs = [torch.empty_like(h) for _ in range(n_ffn)]
            reqs += [dist.irecv(o, src=e, tag=t_combine) for o, e in zip(outs, range(n_attn, world))]
            for r in reqs:
                r.wait()
            return torch.stack(outs).sum(0) / n_ffn
        ins = [torch.empty(b, H, dtype=h.dtype, device=h.device) for _ in range(n_attn)]
        reqs = [dist.irecv(x, src=a, tag=t_dispatch) for a, x in zip(range(n_attn), ins)]
        for r in reqs:
            r.wait()
        reqs = [dist.isend((x @ w).contiguous(), dst=a, tag=t_combine) for a, x in zip(range(n_attn), ins)]
        for r in reqs:
            r.wait()
        return h
    return layer


# ---------------------------------------------------------------- worker
def run_worker(args):
    import torch
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    n_attn = args.n_attn
    attn = rank < n_attn
    res = {"rank": rank, "role": "attention" if attn else "expert", "design": args.design,
           "n_attn": n_attn, "world": world, "points": {}}
    out_path = Path(args.out) / f"{args.design}_a{n_attn}_rank{rank}.json"

    def mark(stage):
        # Written after every stage, so a hang still shows how far each rank got.
        res["last_stage"] = stage
        out_path.write_text(json.dumps(res, indent=1))
        print(f"STAGE rank={rank} {args.design} A={n_attn}: {stage}", flush=True)

    try:
        if args.platform == "tpu":
            import torch_tpu  # noqa: F401
            from torch_tpu._internal import compile as tpu_compile

            dist.init_process_group(backend="tpu_dist")
            device = torch.device("tpu")
            backend = tpu_compile.TpuBackend()
            dtype = torch.bfloat16
        else:
            dist.init_process_group(backend="gloo")
            device = torch.device("cpu")
            backend = "aot_eager"
            dtype = torch.float32
        group = dist.group.WORLD
        torch.empty(1, device=device).cpu()
        mark("initialized")
        if os.environ.get("AFD_TEST_HANG") == args.design and rank == 0:
            time.sleep(10**6)  # test hook: simulate a hang
        # AFD_CONTROL=1 gives expert ranks the wrong weights: every point must
        # then fail the correctness check (proves the check can fail).
        seed_rank = rank + (100 if os.environ.get("AFD_CONTROL") == "1" else 0)
        w = expert_weight(seed_rank, torch).to(dtype).to(device) if not attn else \
            torch.zeros(H, H, dtype=dtype, device=device)

        for b in args.tokens:
            point = {}
            h0 = (attn_input(rank, b, torch) if attn else torch.zeros(b, H)).to(dtype).to(device)

            def chain(k, layer):
                def run(h, *extra):
                    for _ in range(k):
                        h = layer(h, *extra)
                    return h
                return run

            if args.design == "p2p":
                layer = p2p_layer(rank, world, n_attn, b, w, torch)
                runner = lambda k: chain(k, layer)  # noqa: E731
                extra = ()
            else:
                layer = make_layer(args.design, rank, world, n_attn, b, w, group, torch)
                extra = tuple(t.to(device) for t in getattr(layer, "extra", ()))

                def runner(k):
                    torch._dynamo.reset()
                    return torch.compile(chain(k, layer), fullgraph=True, dynamic=False,
                                         backend=backend)

            # Correctness: one layer vs float32 reference.
            one = runner(1)
            mark(f"B={b} compiled+running 1 layer")
            out = one(h0, *extra).cpu().float()
            mark(f"B={b} 1 layer done")
            if attn:
                ref = sum(attn_input(rank, b, torch) @ expert_weight(e, torch)
                          for e in range(n_attn, world)) / (world - n_attn)
                err = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
                point["rel_err"] = err
                point["correct"] = err < 3e-2
            res["points"][str(b)] = dict(point)

            # Timing.
            times = {}
            for k in (1, 1 + REPS):
                fn = one if k == 1 else runner(k)
                mark(f"B={b} timing k={k}")
                for _ in range(2):
                    fn(h0, *extra).cpu()
                samples = []
                for _ in range(5):
                    if os.environ.get("AFD_BARRIER") == "1":
                        dist.barrier()
                    t0 = time.perf_counter()
                    fn(h0, *extra).cpu()
                    samples.append(time.perf_counter() - t0)
                times[k] = statistics.median(samples)
            point["per_layer_us"] = (times[1 + REPS] - times[1]) / REPS * 1e6
            point["single_call_us"] = times[1] * 1e6
            res["points"][str(b)] = point
            mark(f"B={b} done")
            if rank == 0:
                print(f"POINT {args.design} A={n_attn} B={b}: {point}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        res["error"] = repr(exc)[:1500]
    finally:
        out_path.write_text(json.dumps(res, indent=1))
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
    if "error" in res:
        sys.exit(1)


# ---------------------------------------------------------------- launcher
def launch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.platform == "tpu":
        import torch

        for k in ("TORCH_TPU_XPROF_SESSION_ID", "TORCH_TPU_SLICEBUILDER_ADDRESSES",
                  "TORCH_TPU_TOPOLOGY"):
            os.environ.pop(k, None)
        torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=args.world)
        env = os.environ.copy()
    summary = []
    for n_attn in args.splits:
        for design in args.designs:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", f"--nproc-per-node={args.world}", __file__, "--worker",
                   "--platform", args.platform, "--design", design, "--n-attn", str(n_attn),
                   "--tokens", ",".join(map(str, args.tokens)), "--out", str(out)]
            t0 = time.time()
            print(f"== {design} split {n_attn}:{args.world - n_attn}", flush=True)
            log_path = out / f"{design}_a{n_attn}.log"
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    rc = proc.wait(timeout=args.timeout)
                    status = "ok" if rc == 0 else f"exit {rc}"
                except subprocess.TimeoutExpired:
                    status = "TIMEOUT (likely hang)"
                # Kill the whole group, workers included, so no process keeps
                # the TPU devices open for the next design.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                # torchrun puts workers in their own process groups, so also
                # kill them by command line ("--" so pkill does not read the
                # pattern as its own options).
                subprocess.run(["pkill", "-9", "-f", "--", f"{__file__} --worker"], check=False)
                time.sleep(5)
            log = log_path.read_text(errors="replace")
            ranks = []
            for r in range(args.world):
                f = out / f"{design}_a{n_attn}_rank{r}.json"
                ranks.append(json.loads(f.read_text()) if f.exists() else {"rank": r, "error": "no result"})
            row = {"design": design, "split": f"{n_attn}:{args.world - n_attn}",
                   "status": status, "seconds": round(time.time() - t0),
                   "last_stage": sorted({x.get("last_stage", "none") for x in ranks}),
                   "errors": sorted({x["error"][:300] for x in ranks if "error" in x}),
                   "points": {}}
            for b in args.tokens:
                # Points may be partial (a rank hung before timing them).
                pts = [x["points"][str(b)] for x in ranks if str(b) in x.get("points", {})]
                timed = [p for p in pts if "per_layer_us" in p]
                checked = [p for p in pts if "correct" in p]
                timed_attn = [p for p in timed if "correct" in p]
                if pts:
                    row["points"][b] = {
                        "per_layer_us_max": max(p["per_layer_us"] for p in timed) if timed else None,
                        "per_layer_us_attn": statistics.mean(p["per_layer_us"] for p in timed_attn) if timed_attn else None,
                        "correct": all(p["correct"] for p in checked) if checked else None,
                        "rel_err_max": max(p["rel_err"] for p in checked) if checked else None,
                    }
            summary.append(row)
            print(f"   {status} in {row['seconds']}s; last_stage={row['last_stage']}; errors={row['errors'][:1]}", flush=True)
            for b, pt in row["points"].items():
                print(f"   B={b}: {pt}", flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--platform", choices=("tpu", "cpu"), default="tpu")
    ap.add_argument("--design")
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--n-attn", type=int, default=4)
    ap.add_argument("--splits", default="4,2")
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--tokens", default="1,8,32,128")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.tokens = [int(t) for t in str(args.tokens).split(",") if t]
    if args.worker:
        run_worker(args)
    else:
        args.designs = [d for d in args.designs.split(",") if d]
        args.splits = [int(s) for s in args.splits.split(",") if s]
        launch(args)


if __name__ == "__main__":
    main()
