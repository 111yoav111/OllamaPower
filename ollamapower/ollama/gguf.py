"""Read what split mode needs from a GGUF file's header (never the weights).

The split planner has to know how big each layer is to put the right number of
layers on each GPU: llama.cpp's ``rpc-server`` has no memory limit of its own,
so a helper's VRAM cap is enforced by giving it only as many layers as fit.

This is a small, dependency-free reader rather than the ``gguf`` package
(which pulls in numpy). It reads the key/value metadata and the tensor table,
and derives each tensor's size from the gap to the next tensor's offset, so it
needs no table of quantisation block sizes. The file is untrusted input:
every count and length is bounded before anything is allocated.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
MAX_KV = 1_000_000
MAX_TENSORS = 1_000_000
MAX_STRING = 16 * 1024 * 1024
MAX_ARRAY = 64 * 1024 * 1024
MAX_DIMS = 8
DEFAULT_ALIGNMENT = 32
# Arrays longer than this (token vocabularies) are skipped, not kept.
KEEP_ARRAY = 1024

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_SCALARS = {
    U8: "<B",
    I8: "<b",
    U16: "<H",
    I16: "<h",
    U32: "<I",
    I32: "<i",
    F32: "<f",
    BOOL: "<?",
    U64: "<Q",
    I64: "<q",
    F64: "<d",
}
_BLOCK = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    """Not a GGUF file, or one we can't use."""


@dataclass
class GGUFInfo:
    path: Path
    file_size: int
    version: int
    architecture: str
    block_count: int
    context_length: int = 0
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    split_count: int = 1  # >1: the model is sharded over several files
    layer_bytes: list[int] = field(default_factory=list)  # weights of blk.0 .. blk.N-1
    output_bytes: int = 0  # the output layer (llama.cpp's "layer N")
    input_bytes: int = 0  # token embeddings: always kept on the hub's CPU
    other_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weights_bytes(self) -> int:
        return sum(self.layer_bytes) + self.output_bytes + self.input_bytes + self.other_bytes

    def kv_bytes_per_layer(self, ctx: int) -> int:
        """KV cache for one layer at ``ctx`` tokens (f16), an upper bound."""
        heads_kv = self.head_count_kv or self.head_count
        k = self.key_length or (self.embedding_length // self.head_count if self.head_count else 0)
        v = self.value_length or k
        return ctx * heads_kv * (k + v) * 2


class _Reader:
    def __init__(self, fh: BinaryIO, size: int) -> None:
        self.fh = fh
        self.size = size

    def read(self, n: int) -> bytes:
        if n < 0 or self.fh.tell() + n > self.size:
            raise GGUFError("truncated GGUF header")
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError("truncated GGUF header")
        return data

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > MAX_STRING:
            raise GGUFError("string too long in GGUF header")
        return self.read(n).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        n = self.unpack("<Q")
        if n > MAX_STRING:
            raise GGUFError("string too long in GGUF header")
        self.fh.seek(n, os.SEEK_CUR)
        if self.fh.tell() > self.size:
            raise GGUFError("truncated GGUF header")

    def value(self, kind: int) -> Any:
        if kind in _SCALARS:
            return self.unpack(_SCALARS[kind])
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item = self.unpack("<I")
            count = self.unpack("<Q")
            if count > MAX_ARRAY or item == ARRAY:
                raise GGUFError("array too long or nested in GGUF header")
            if count > KEEP_ARRAY:
                self._skip_array(item, count)
                return None
            return [self.value(item) for _ in range(count)]
        raise GGUFError(f"unknown GGUF value type {kind}")

    def _skip_array(self, item: int, count: int) -> None:
        if item in _SCALARS:
            self.fh.seek(struct.calcsize(_SCALARS[item]) * count, os.SEEK_CUR)
            if self.fh.tell() > self.size:
                raise GGUFError("truncated GGUF header")
        elif item == STRING:
            for _ in range(count):
                self.skip_string()
        else:
            raise GGUFError(f"unknown GGUF value type {item}")


def _int(value: Any) -> int:
    if isinstance(value, list):  # per-layer values (e.g. head_count_kv): plan for the largest
        value = max((v for v in value if isinstance(v, int)), default=0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def read_gguf(path: Path) -> GGUFInfo:
    """Parse ``path``'s header. Raises ``GGUFError`` (or ``OSError`` if unreadable)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        r = _Reader(fh, size)
        if r.read(4) != MAGIC:
            raise GGUFError(f"{path.name} is not a GGUF file")
        version = r.unpack("<I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        if n_tensors > MAX_TENSORS or n_kv > MAX_KV:
            raise GGUFError("implausible GGUF header")
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        tensors: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = r.string()
            dims = r.unpack("<I")
            if dims > MAX_DIMS:
                raise GGUFError("too many tensor dimensions")
            r.read(8 * dims)
            r.unpack("<I")  # type
            tensors.append((name, r.unpack("<Q")))
        alignment = _int(meta.get("general.alignment")) or DEFAULT_ALIGNMENT
        data_start = -(-fh.tell() // alignment) * alignment
    data_size = size - data_start
    if data_size < 0:
        raise GGUFError("truncated GGUF file")

    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise GGUFError("GGUF file has no general.architecture")

    def arch_key(name: str) -> int:
        return _int(meta.get(f"{arch}.{name}"))

    info = GGUFInfo(
        path=path,
        file_size=size,
        version=version,
        architecture=arch,
        block_count=arch_key("block_count"),
        context_length=arch_key("context_length"),
        embedding_length=arch_key("embedding_length"),
        head_count=arch_key("attention.head_count"),
        head_count_kv=arch_key("attention.head_count_kv"),
        key_length=arch_key("attention.key_length"),
        value_length=arch_key("attention.value_length"),
        split_count=max(1, _int(meta.get("split.count"))),
        metadata={k: v for k, v in meta.items() if not isinstance(v, list) and not k.startswith("tokenizer.")},
    )

    ordered = sorted(tensors, key=lambda t: t[1])
    layers = [0] * info.block_count
    token_embd = output = 0
    for i, (name, offset) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else data_size
        nbytes = end - offset
        if offset > data_size or nbytes < 0:
            raise GGUFError("GGUF tensor offsets are out of range")
        block = _BLOCK.match(name)
        if block and int(block.group(1)) < info.block_count:
            layers[int(block.group(1))] += nbytes
        elif name.startswith("token_embd."):
            token_embd += nbytes
        elif name.startswith("output.") or name.startswith("output_norm."):
            output += nbytes
        else:
            info.other_bytes += nbytes
    info.layer_bytes = layers
    info.input_bytes = token_embd
    # Tied embeddings: llama.cpp puts a copy of token_embd on the output layer's device.
    has_output_weight = any(n == "output.weight" for n, _ in tensors)
    info.output_bytes = output + (0 if has_output_weight else token_embd)
    return info
