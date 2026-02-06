from __future__ import annotations


def _load_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def is_cuda_available() -> bool:
    torch = _load_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_mps_available() -> bool:
    torch = _load_torch()
    if torch is None:
        return False
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False


def resolve_device(preferred: str | None) -> str:
    choice = (preferred or "auto").lower()
    if choice == "auto":
        if is_cuda_available():
            return "cuda"
        if is_mps_available():
            return "mps"
        return "cpu"
    if choice == "cuda":
        return "cuda" if is_cuda_available() else "cpu"
    if choice == "mps":
        return "mps" if is_mps_available() else "cpu"
    if choice == "cpu":
        return "cpu"
    raise ValueError("device must be one of auto/cuda/mps/cpu")
