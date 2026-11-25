"""Lossless compression helpers for graph edge payloads."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionStats:
    original_bytes: int
    compressed_bytes: int

    @property
    def ratio(self) -> float:
        if self.original_bytes == 0:
            return 1.0
        return self.compressed_bytes / self.original_bytes


class EdgeCodec:
    """Serialize and compress edges without changing their order or values."""

    @staticmethod
    def encode(edges: tuple[tuple[int, int], ...]) -> tuple[bytes, CompressionStats]:
        raw = json.dumps(edges, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(raw)
        return compressed, CompressionStats(len(raw), len(compressed))

    @staticmethod
    def decode(payload: bytes) -> tuple[tuple[int, int], ...]:
        decoded = json.loads(zlib.decompress(payload).decode("utf-8"))
        if not isinstance(decoded, list):
            raise ValueError("compressed edge payload must contain a list")
        return tuple((int(source), int(target)) for source, target in decoded)
