#!/usr/bin/env python3
"""AFD with per-role branches in JAX: one program over all chips, where each
chip executes only its own role's compute (lax.cond on the device index
inside shard_map) and the two all-to-all exchanges sit outside the branches.

This is the shape of an AFD layer that vllm-torchtpu could call as a JAX
custom op (it already runs shard_map ops this way, e.g. pcp_streaming_rpa).

Per layer, A attention chips and F = N - A expert chips:
  attention: dispatch its B token rows to every expert chip; combine: sum
             the expert chips' results.
  expert:    multiply received rows by its own weight (a stand-in for its
             share of the experts) and send them back.

Checks:
  correct   one layer vs a float32 numpy reference (error < 3%).
  skip      each role also has K matmuls of side work. "both" runs both
            roles' work on every chip (jnp.where); "branch" runs only the
            chip's own work (lax.cond). If the branch really skips the other
            role, "branch" is about K matmuls faster per layer.

  python3 afd/jax_role_test.py --out DIR                 # TPU: all local chips
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
    python3 afd/jax_role_test.py --out DIR --cpu        # logic check
"""
import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np

H = 2048
SEED = 1234
REPS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--splits", default="4,2")
    ap.add_argument("--tokens", default="1,8,32,128")
    ap.add_argument("--work", type=int, default=40, help="side-work matmuls per role")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    try:
        from jax import shard_map
    except ImportError:  # older JAX
        from jax.experimental.shard_map import shard_map

    devices = jax.devices("cpu" if args.cpu else None)
    n = len(devices)
    mesh = Mesh(np.array(devices), ("x",))
    dtype = jnp.float32 if args.cpu else jnp.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{n} devices: {devices[0].device_kind}", flush=True)

    rng = np.random.default_rng(SEED)
    w_all = rng.standard_normal((n, H, H), dtype=np.float32) / H**0.5
    results = []

    def vary(x):
        # Mark a per-chip value as varying over the mesh axis, so both cond
        # branches have the same type inside shard_map.
        return lax.pcast(x, ("x",), to="varying")

    def a2a(x):
        return lax.all_to_all(x, "x", 0, 0, tiled=True)

    for n_attn in [int(s) for s in args.splits.split(",")]:
        n_ffn = n - n_attn
        w_np = w_all.copy()
        w_np[:n_attn] = 0  # attention chips hold no expert weights
        for b in [int(t) for t in args.tokens.split(",")]:
            h_np = rng.standard_normal((n, b, H), dtype=np.float32)
            h_np[n_attn:] = 0
            ref = np.stack([
                sum(h_np[a] @ w_all[e] for e in range(n_attn, n)) / n_ffn
                for a in range(n_attn)])

            def one_layer(h, w, z, mode):
                """h: (b,H) this chip's tokens; w: (H,H); z: (b,H) side work."""
                i = lax.axis_index("x")
                is_attn = i < n_attn
                dst_ffn = (jnp.arange(n) >= n_attn).astype(h.dtype)

                def work(v):
                    for _ in range(args.work):
                        v = v + (v @ w) * 1e-30
                    return v

                if mode == "both":
                    z = jnp.where(is_attn, work(z), work(z * 2))
                else:
                    z = lax.cond(is_attn, work, lambda v: work(v * 2), z)

                send = lax.cond(
                    is_attn,
                    lambda hh: (hh[None] * dst_ffn[:, None, None]).reshape(n * b, H),
                    lambda hh: vary(jnp.zeros((n * b, H), hh.dtype)), h)
                recv = a2a(send)
                y = lax.cond(is_attn, lambda r: jnp.zeros_like(r), lambda r: r @ w, recv)
                back = a2a(y)
                out = lax.cond(
                    is_attn,
                    lambda bk, hh: (bk.reshape(n, b, H) * dst_ffn[:, None, None]).sum(0) / n_ffn,
                    lambda bk, hh: hh, back, h)
                return out + z[:1, :1] * 1e-30, z

            def build(k, mode):
                def per_chip(h, w, z):
                    h, w, z = h[0], w[0], z[0]
                    for _ in range(k):
                        h, z = one_layer(h, w, z, mode)
                    return h[None]
                f = shard_map(per_chip, mesh=mesh, in_specs=(P("x"), P("x"), P("x")),
                              out_specs=P("x"))
                return jax.jit(f)

            sh = NamedSharding(mesh, P("x"))
            h = jax.device_put(jnp.asarray(h_np, dtype), sh)
            w = jax.device_put(jnp.asarray(w_np, dtype), sh)
            z = jax.device_put(jnp.asarray(rng.standard_normal((n, b, H)) * 0.01, dtype), sh)

            row = {"split": f"{n_attn}:{n_ffn}", "tokens": b}
            for mode in ("branch", "both"):
                try:
                    f1 = build(1, mode)
                    out = np.asarray(f1(h, w, z), dtype=np.float32)[:n_attn]
                    err = float(np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-9))
                    row[f"{mode}_rel_err"] = err
                    row[f"{mode}_correct"] = err < 3e-2
                    times = {}
                    for k in (1, 1 + REPS):
                        f = f1 if k == 1 else build(k, mode)
                        for _ in range(2):
                            f(h, w, z).block_until_ready()
                        samples = []
                        for _ in range(7):
                            t0 = time.perf_counter()
                            f(h, w, z).block_until_ready()
                            samples.append(time.perf_counter() - t0)
                        times[k] = statistics.median(samples)
                    row[f"{mode}_per_layer_us"] = (times[1 + REPS] - times[1]) / REPS * 1e6
                except Exception as exc:  # noqa: BLE001
                    row[f"{mode}_error"] = repr(exc)[:600]
            results.append(row)
            print("ROW", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in row.items()}), flush=True)
            (out_dir / "jax_role_summary.json").write_text(json.dumps(results, indent=1))

    # Exchange-only cost: the same layer with no side work.
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
