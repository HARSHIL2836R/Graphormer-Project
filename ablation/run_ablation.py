"""
Inference ablation driver for Graphormer.

Measures average single-pass forward latency and peak GPU memory across
attention-backend / precision combinations on synthetic molecular graphs that
match the ZINC / OGBG-MolHIV input schema produced by Microsoft Graphormer's
collator.

Examples
--------
# Random-init slim model on CPU (sanity):
python run_ablation.py --model slim --device cpu

# Full ablation on CUDA with the base architecture:
python run_ablation.py --model base --device cuda \
       --backends manual sdpa sdpa-flash --dtypes fp32 fp16 bf16 \
       --batch-size 4 --num-nodes 32 --iters 50 --warmup 10

# Load a real Microsoft checkpoint:
python run_ablation.py --model base --checkpoint checkpoint_best_pcqm4mv1.pt \
       --backends manual sdpa --iters 30

KV caching: explicitly NOT included. See note printed at startup.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import nullcontext
from typing import Optional

import torch

from graphormer_inference import (
    GraphormerConfig,
    GraphormerForGraphPrediction,
    count_parameters,
    load_pretrained_checkpoint,
)

KV_CACHE_NOTE = (
    "KV cache: NOT APPLICABLE. Graphormer is encoder-only and processes the\n"
    "entire padded graph [CLS, v1..vN] in a single bidirectional pass. There is\n"
    "no autoregressive step, and the attention bias depends on all node-pairs\n"
    "jointly, so K/V are produced and consumed exactly once per forward.\n"
    "KV caching only accelerates incremental decoding and is excluded here."
)


# ---------------------------------------------------------------------------
# Synthetic molecular graph batch (mimics graphormer/data/collator.py output)
# ---------------------------------------------------------------------------
def make_dummy_batch(
    batch_size: int,
    num_nodes: int,
    num_atom_features: int = 9,
    num_edge_features: int = 3,
    multi_hop_max_dist: int = 5,
    num_atoms_vocab: int = 512 * 9,
    num_edges_vocab: int = 512 * 3,
    num_in_degree: int = 512,
    num_out_degree: int = 512,
    num_spatial: int = 512,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Synthesise a batch with the same key set as Graphormer's collator.

    Returned tensors:
        x              : [B, T, K]            atom-feature ids (>=1; 0 == pad)
        in_degree      : [B, T]               in-degree ids
        out_degree     : [B, T]               out-degree ids
        attn_bias      : [B, T+1, T+1]        base bias (zeros, used by collator)
        spatial_pos    : [B, T, T]            shortest-path distance ids
        attn_edge_type : [B, T, T, K_e]       edge-type ids per node-pair
        edge_input     : [B, T, T, D_max, K_e] multi-hop edge features along SP

    All ids are kept inside the embedding-table ranges declared by
    GraphormerConfig so embedding lookups never go out of bounds.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    x = torch.randint(1, num_atoms_vocab, (batch_size, num_nodes, num_atom_features), generator=g)
    in_degree = torch.randint(1, num_in_degree, (batch_size, num_nodes), generator=g)
    out_degree = torch.randint(1, num_out_degree, (batch_size, num_nodes), generator=g)

    attn_bias = torch.zeros(batch_size, num_nodes + 1, num_nodes + 1)
    spatial_pos = torch.randint(1, min(num_spatial, num_nodes + 1), (batch_size, num_nodes, num_nodes), generator=g)
    attn_edge_type = torch.randint(
        1, num_edges_vocab, (batch_size, num_nodes, num_nodes, num_edge_features), generator=g
    )
    edge_input = torch.randint(
        1,
        num_edges_vocab,
        (batch_size, num_nodes, num_nodes, multi_hop_max_dist, num_edge_features),
        generator=g,
    )

    batch = {
        "x": x,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "attn_bias": attn_bias,
        "spatial_pos": spatial_pos,
        "attn_edge_type": attn_edge_type,
        "edge_input": edge_input,
    }
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def _autocast_ctx(device: str, dtype_name: str):
    if dtype_name == "fp32":
        return nullcontext()
    if device != "cuda":
        return nullcontext()
    if dtype_name == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if dtype_name == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(dtype_name)


@torch.no_grad()
def profile_run(
    cfg: GraphormerConfig,
    backend: str,
    dtype_name: str,
    batch: dict,
    device: str,
    iters: int,
    warmup: int,
    checkpoint: Optional[str] = None,
) -> dict:
    """Build a fresh model with the requested backend, profile a forward loop."""
    torch.manual_seed(0)
    model = GraphormerForGraphPrediction(cfg, attn_backend=backend).to(device).eval()
    if checkpoint:
        load_pretrained_checkpoint(model, checkpoint, strict=False, verbose=False)

    # Reset CUDA memory stats and warm up.
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    autocast = _autocast_ctx(device, dtype_name)

    try:
        with autocast:
            for _ in range(warmup):
                _ = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        with autocast:
            for _ in range(iters):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else float("nan")
        )
        ok = True
        err = None
    except Exception as e:  # noqa: BLE001
        latencies_ms = []
        peak_mem_mb = float("nan")
        ok = False
        err = f"{type(e).__name__}: {e}"

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not ok:
        return {
            "backend": backend,
            "dtype": dtype_name,
            "ok": False,
            "error": err,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "std_ms": None,
            "peak_mem_mb": None,
        }

    arr = torch.tensor(latencies_ms)
    return {
        "backend": backend,
        "dtype": dtype_name,
        "ok": True,
        "mean_ms": float(arr.mean()),
        "median_ms": float(arr.median()),
        "p90_ms": float(arr.quantile(0.9)),
        "std_ms": float(arr.std(unbiased=False)),
        "peak_mem_mb": peak_mem_mb,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _make_cfg(name: str) -> GraphormerConfig:
    name = name.lower()
    if name == "base":
        return GraphormerConfig.graphormer_base()
    if name == "slim":
        return GraphormerConfig.graphormer_slim()
    if name == "large":
        return GraphormerConfig.graphormer_large()
    raise ValueError(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graphormer inference ablation")
    p.add_argument("--model", choices=["slim", "base", "large"], default="slim")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a Microsoft Graphormer .pt checkpoint (optional)")
    p.add_argument("--device", choices=["cpu", "cuda"],
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-nodes", type=int, default=32)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--backends", nargs="+",
                   default=["manual", "sdpa", "sdpa-flash"],
                   choices=["manual", "sdpa", "sdpa-flash"])
    p.add_argument("--dtypes", nargs="+", default=["fp32"],
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--json-out", type=str, default=None,
                   help="Optional path to dump raw results as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _make_cfg(args.model)

    print("=" * 78)
    print(f" Graphormer inference ablation  |  device={args.device}  |  model={args.model}")
    print("=" * 78)
    print(KV_CACHE_NOTE)
    print("-" * 78)

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
    print(f"PyTorch: {torch.__version__}    CUDA build: {torch.version.cuda}")
    print(
        f"Architecture: layers={cfg.encoder_layers} dim={cfg.encoder_embed_dim} "
        f"heads={cfg.encoder_attention_heads} ffn={cfg.encoder_ffn_embed_dim}"
    )
    # Build once just to count params.
    _tmp = GraphormerForGraphPrediction(cfg, attn_backend="manual")
    print(f"Parameter count: {count_parameters(_tmp) / 1e6:.2f} M")
    del _tmp
    gc.collect()

    print(
        f"Batch: B={args.batch_size}  T={args.num_nodes}  "
        f"iters={args.iters}  warmup={args.warmup}"
    )
    print("-" * 78)

    batch = make_dummy_batch(
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        multi_hop_max_dist=cfg.multi_hop_max_dist,
        num_atoms_vocab=cfg.num_atoms,
        num_edges_vocab=cfg.num_edges,
        num_in_degree=cfg.num_in_degree,
        num_out_degree=cfg.num_out_degree,
        num_spatial=cfg.num_spatial,
        device=args.device,
    )

    results = []
    header = f"{'backend':<14}{'dtype':<8}{'mean ms':>10}{'median ms':>12}{'p90 ms':>10}{'std ms':>10}{'peak MB':>11}  status"
    print(header)
    print("-" * len(header))

    for backend in args.backends:
        for dtype_name in args.dtypes:
            r = profile_run(
                cfg=cfg,
                backend=backend,
                dtype_name=dtype_name,
                batch=batch,
                device=args.device,
                iters=args.iters,
                warmup=args.warmup,
                checkpoint=args.checkpoint,
            )
            results.append(r)
            if r["ok"]:
                print(
                    f"{r['backend']:<14}{r['dtype']:<8}"
                    f"{r['mean_ms']:>10.3f}{r['median_ms']:>12.3f}"
                    f"{r['p90_ms']:>10.3f}{r['std_ms']:>10.3f}"
                    f"{r['peak_mem_mb']:>11.2f}  OK"
                )
            else:
                print(f"{r['backend']:<14}{r['dtype']:<8}  FAILED  {r['error']}")

    print("-" * len(header))
    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        baseline = next((r for r in ok_results if r["backend"] == "manual" and r["dtype"] == "fp32"), ok_results[0])
        print(
            f"Speedup vs. {baseline['backend']}/{baseline['dtype']} "
            f"({baseline['mean_ms']:.3f} ms):"
        )
        for r in ok_results:
            speedup = baseline["mean_ms"] / r["mean_ms"]
            mem_ratio = (
                r["peak_mem_mb"] / baseline["peak_mem_mb"]
                if baseline["peak_mem_mb"] and r["peak_mem_mb"] else float("nan")
            )
            print(
                f"  {r['backend']:<14}{r['dtype']:<6}"
                f"latency x{speedup:5.2f}    peak-mem x{mem_ratio:5.2f}"
            )

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {
                    "config": vars(args),
                    "arch": cfg.__dict__,
                    "results": results,
                    "kv_cache_note": KV_CACHE_NOTE,
                },
                fh,
                indent=2,
            )
        print(f"[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
