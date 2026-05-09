"""Download and load Microsoft's released Graphormer checkpoints."""

from __future__ import annotations

import io
import pickle as _pickle
from pathlib import Path
from typing import Optional

import torch
import torch.hub

from .model import GraphormerConfig, MiniGraphormer, graphormer_base_pcqm4mv2_config


# ---------------------------------------------------------------------------
# Forgiving unpickler.
# ---------------------------------------------------------------------------
# Microsoft's released checkpoints were saved by fairseq + omegaconf, so the
# pickle inside the .pt file references classes like
# `fairseq.dataclass.configs.FairseqConfig` and `omegaconf.dictconfig.DictConfig`.
# We don't have (or need) those packages — we only consume the `"model"` key
# (a flat dict of tensors). The unpickler below substitutes a no-op stub for
# any class it can't import, so the metadata becomes harmless placeholders
# while every tensor loads normally.
#
# This is safe because (a) we trust the source, (b) PyTorch's tensor loading
# happens through its own zip+storage mechanism, not through this pickle path.
# ---------------------------------------------------------------------------


class _Stub:
    """No-op placeholder used when a pickled class isn't importable."""
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): pass
    def __reduce__(self): return (_Stub, ())
    def __repr__(self): return "<_Stub>"


class _ForgivingUnpickler(_pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return _Stub


class _ForgivingPickle:
    """Adapter that satisfies torch.load's `pickle_module=` interface."""
    Unpickler = _ForgivingUnpickler
    HIGHEST_PROTOCOL = _pickle.HIGHEST_PROTOCOL

    @staticmethod
    def load(f, **kw):
        return _ForgivingUnpickler(f, **kw).load()

    @staticmethod
    def loads(s, **kw):
        return _ForgivingUnpickler(io.BytesIO(s), **kw).load()


def _safe_torch_load(path: Path):
    """torch.load that tolerates checkpoints saved with non-installed classes."""
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,           # released ckpts contain non-tensor metadata
        pickle_module=_ForgivingPickle,
    )


# Each name maps to an ordered list of mirrors. The fetcher tries them in
# order and stops at the first success. Microsoft's original `ml2md.blob`
# host went offline; the Zenodo record below (from the MassFormer authors)
# is a community-hosted, byte-identical copy of `checkpoint_best_pcqm4mv2.pt`
# (193 MB).
PRETRAINED_URLS = {
    "pcqm4mv2_graphormer_base": [
        "https://zenodo.org/records/8399738/files/checkpoint_best_pcqm4mv2.pt?download=1",
        "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt",
    ],
    # The v1 / molhiv checkpoints were also on the now-dead ml2md host; left
    # here as references in case Microsoft re-publishes them.
    "pcqm4mv1_graphormer_base": [
        "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv1.pt",
    ],
    "pcqm4mv1_graphormer_base_for_molhiv": [
        "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
    ],
}


def _purge_if_corrupt(cache_path: Path) -> None:
    """Delete the cached file ONLY if our forgiving loader also fails.

    torch.hub.download_url_to_file skips the download when the target file
    already exists, so a partial / interrupted prior download would otherwise
    poison every subsequent call. But "torch.load failed" alone is not enough
    evidence of corruption — a perfectly good fairseq checkpoint will trip
    torch.load with ModuleNotFoundError. So we use the forgiving loader as
    the ground truth: if even that fails, the bytes themselves are bad.
    """
    if not cache_path.exists():
        return
    try:
        _safe_torch_load(cache_path)
    except Exception as e:  # noqa: BLE001
        print(f"[mini_graphormer] cached file is unreadable ({type(e).__name__}); "
              f"deleting and re-downloading: {cache_path}")
        cache_path.unlink()


def _fetch_state_dict(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if p.exists():
        ckpt = _safe_torch_load(p)
    elif name_or_path in PRETRAINED_URLS:
        urls = PRETRAINED_URLS[name_or_path]
        cached_name = f"{name_or_path}.pt"
        cache_path = Path(torch.hub.get_dir()) / "checkpoints" / cached_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Heal stale corrupt cache from a prior interrupted download.
        _purge_if_corrupt(cache_path)

        last_err: Optional[Exception] = None
        ckpt = None
        for url in urls:
            try:
                # Download separately from load so we can supply our own
                # pickle_module to torch.load (load_state_dict_from_url has
                # no parameter for that).
                if not cache_path.exists():
                    print(f"[mini_graphormer] downloading {url}")
                    torch.hub.download_url_to_file(url, str(cache_path), progress=True)
                ckpt = _safe_torch_load(cache_path)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[mini_graphormer] mirror failed ({type(e).__name__}): {url}")
                last_err = e
                # Only purge if the bytes themselves are bad (not just because
                # of unimportable metadata classes — those are stubbed by the
                # forgiving unpickler).
                _purge_if_corrupt(cache_path)
        if ckpt is None:
            raise RuntimeError(f"all mirrors for '{name_or_path}' failed") from last_err
    else:
        raise FileNotFoundError(
            f"'{name_or_path}' is neither a local checkpoint path nor a known release "
            f"({sorted(PRETRAINED_URLS)})"
        )
    # Released checkpoints are wrapped under "model".
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def load_pretrained(
    name_or_path: str = "pcqm4mv2_graphormer_base",
    cfg: Optional[GraphormerConfig] = None,
    strict: bool = True,
) -> MiniGraphormer:
    """Build a `MiniGraphormer` and load the requested weights into it.

    Args:
        name_or_path: a key in `PRETRAINED_URLS` or a path to a local .pt file.
        cfg: optional override. If None, defaults to the PCQM4Mv2 base config
             (12 / 768 / 32 heads, GeLU, post-LN), which matches the released
             v1/v2 base checkpoints. For the molhiv pre-LN checkpoint, pass a
             config with `pre_layernorm=True`.
        strict: forwarded to `load_state_dict`.
    """
    if cfg is None:
        cfg = graphormer_base_pcqm4mv2_config()
        if name_or_path == "pcqm4mv1_graphormer_base_for_molhiv":
            cfg.pre_layernorm = True

    model = MiniGraphormer(cfg)
    state = _fetch_state_dict(name_or_path)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        # In strict=False mode, surface what differed so ablations can be debugged.
        print(f"[mini_graphormer] missing={missing}\nunexpected={unexpected}")
    model.eval()
    return model
