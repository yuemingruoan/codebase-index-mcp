from __future__ import annotations


def is_text_bytes(sample: bytes, *, threshold: float = 0.3) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    control = 0
    for byte in sample:
        if byte < 32 and byte not in (9, 10, 13, 12, 8):
            control += 1
    return (control / len(sample)) <= threshold


def is_text_file(path: str, *, sample_size: int = 4096) -> bool:
    with open(path, "rb") as handle:
        sample = handle.read(sample_size)
    return is_text_bytes(sample)
