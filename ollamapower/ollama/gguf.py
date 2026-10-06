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
MAX_BLOCKS = 4096  # the largest real models have ~200 layers
MAX_DIMENSION = 1 << 26  # context length, widths and head counts beyond this are corrupt or hostile
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
_NEXTN = re.compile(r"^blk\.(\d+)\.nextn\.")
# Tensors of parts llama.cpp's text model doesn't load from the main GGUF. Ollama's own engine
# keeps a vision/audio encoder or a multi-token-prediction head there; llama.cpp wants the
# encoders in a separate projector file and its MTP layers as declared ``blk.N.nextn.*``.
FOREIGN_PREFIXES = {"v.": "vision", "mm.": "vision", "a.": "audio", "mtp.": "MTP", "mtp_": "MTP"}


# KV cache shapes llama.cpp uses (see llama-hparams.cpp / llama-kv-cache*.cpp at the pinned build).
UBATCH = 512  # llama-server's default -ub: a sliding-window cache keeps the window plus one ubatch
KV_PAD = 256
# Architectures whose sliding-window layers follow a fixed pattern unless the header gives one
# (every n-th layer is global, the others slide), from each model's load_arch_hparams.
SWA_PATTERN_DEFAULTS = {"gemma2": 2, "gemma3": 6, "cohere2": 4, "gpt-oss": 2}
# Hybrid models whose linear-attention layers are all but every n-th one, unless the header says.
RECURRENT_INTERVAL_DEFAULTS = {"qwen3next": 4, "qwen35": 4, "qwen35moe": 4}
PURE_RECURRENT = frozenset({"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"})
PARALLEL_HYBRID = frozenset({"falcon-h1"})  # attention and SSM side by side in every layer
# MLA models whose cache has more than the compressed latent (DSA indexer caches): keep the
# old, larger estimate for those rather than under-estimate.
MLA_WITH_EXTRA_CACHES = frozenset({"deepseek32", "deepseek4", "glm-dsa", "glm5-next"})


class GGUFError(ValueError):
    """Not a GGUF file, or one we can't use."""


@dataclass(frozen=True)
class KVLayer:
    """What one block keeps per sequence besides its weights."""

    per_token: int  # K+V cache bytes per token (f16); 0 for layers without attention
    window: int = 0  # >0: a sliding-window layer, which caches at most this many tokens
    state: int = 0  # recurrent (SSM/RWKV) state in bytes, independent of the context


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
    nextn_layers: int = 0  # trailing blocks that are llama.cpp MTP ("NextN") layers
    foreign: dict[str, int] = field(default_factory=dict)  # part ("vision", "audio", "MTP") -> tensor count
    layer_bytes: list[int] = field(default_factory=list)  # weights of blk.0 .. blk.N-1
    output_bytes: int = 0  # the output layer (llama.cpp's "layer N")
    input_bytes: int = 0  # token embeddings: always kept on the hub's CPU
    other_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    kv_layers: list[KVLayer] = field(default_factory=list)  # per block; empty = every block full attention

    @property
    def weights_bytes(self) -> int:
        return sum(self.layer_bytes) + self.output_bytes + self.input_bytes + self.other_bytes

    def _uniform_kv_per_token(self) -> int:
        heads_kv = self.head_count_kv or self.head_count
        k = self.key_length or (self.embedding_length // self.head_count if self.head_count else 0)
        v = self.value_length or k
        return heads_kv * (k + v) * 2

    def kv_bytes(self, ctx: int) -> list[int]:
        """Cache bytes each block needs at ``ctx`` tokens: KV (f16, windows applied) plus recurrent state."""
        if not self.kv_layers:
            return [ctx * self._uniform_kv_per_token()] * self.block_count
        return [
            layer.per_token * (min(ctx, layer.window) if layer.window else ctx) + layer.state
            for layer in self.kv_layers
        ]

    def kv_bytes_total(self, ctx: int) -> int:
        return sum(self.kv_bytes(ctx))

    def kv_bytes_per_layer(self, ctx: int) -> int:
        """Cache of the largest block at ``ctx`` tokens (for display: an upper bound per layer)."""
        return max(self.kv_bytes(ctx), default=0)


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


def _per_layer(value: Any, n: int) -> list[int] | None:
    """A per-layer header value as ``n`` ints (a scalar repeats), or None if absent or malformed."""
    if isinstance(value, list):
        if len(value) != n or not all(isinstance(v, int) for v in value):
            return None
        return [max(0, int(v)) for v in value]
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return [value] * n
    return None


def _kv_layout(meta: dict[str, Any], info: GGUFInfo) -> list[KVLayer]:
    """Per-block cache shape, as llama.cpp allocates it: MLA, sliding windows, recurrent layers.

    Anything unknown falls back to the full-attention estimate, which is an upper bound.
    """
    arch, n = info.architecture, info.block_count
    trunk = n - info.nextn_layers

    def key(name: str) -> Any:
        return meta.get(f"{arch}.{name}")

    def num(name: str) -> int:
        return min(_int(key(name)), MAX_DIMENSION)

    heads = _per_layer(key("attention.head_count"), n) or [0] * n
    # Absent: as many KV heads as heads. An array may hold 0 for layers without attention.
    explicit_kv = isinstance(key("attention.head_count_kv"), list)
    heads_kv = _per_layer(key("attention.head_count_kv"), n) if key("attention.head_count_kv") else None
    if heads_kv is None or (not explicit_kv and not any(heads_kv)):
        heads_kv = heads
    k_len = info.key_length or (info.embedding_length // info.head_count if info.head_count else 0)
    v_len = info.value_length or k_len
    k_swa = num("attention.key_length_swa") or k_len
    v_swa = num("attention.value_length_swa") or (k_swa if num("attention.key_length_swa") else v_len)

    # MLA (DeepSeek-V2/V3, Kimi K2, ...): one compressed latent plus the RoPE part per token, K only.
    kv_lora, mla = num("attention.kv_lora_rank"), num("attention.key_length_mla")
    mla_per_token = 0
    if kv_lora and mla and arch not in MLA_WITH_EXTRA_CACHES:
        rope = num("rope.dimension_count") or max(0, k_len - kv_lora)
        mla_per_token = (kv_lora + rope) * 2

    # Sliding-window attention: those layers keep the window (plus one ubatch), not the context.
    window = num("attention.sliding_window")
    swa = [False] * n
    if window > 0:
        pattern = key("attention.sliding_window_pattern")
        if isinstance(pattern, list) and len(pattern) == n:
            swa = [bool(x) for x in pattern]
        else:
            every = _int(pattern) if isinstance(pattern, int) and not isinstance(pattern, bool) else None
            every = SWA_PATTERN_DEFAULTS.get(arch) if every is None else every
            if every is not None:
                swa = [i < trunk and (every == 0 or i % every < every - 1) for i in range(n)]
    cells = -(-(window + UBATCH) // KV_PAD) * KV_PAD

    # Recurrent layers (SSM, RWKV, linear attention): a fixed state instead of a KV cache.
    recurrent = [arch in PURE_RECURRENT] * n
    listed = key("attention.recurrent_layers")
    interval = num("full_attention_interval") or RECURRENT_INTERVAL_DEFAULTS.get(arch, 0)
    if isinstance(listed, list) and len(listed) == n:
        recurrent = [bool(x) for x in listed]
    elif interval:
        recurrent = [i < trunk and (i + 1) % interval != 0 for i in range(n)]
    if explicit_kv:
        recurrent = [r or kv == 0 for r, kv in zip(recurrent, heads_kv, strict=True)]
    n_embd = info.embedding_length
    state: int | None = None
    if num("wkv.head_size"):
        state = ((num("token_shift_count") or 2) * n_embd + n_embd * num("wkv.head_size")) * 4
    elif num("shortconv.l_cache"):
        state = n_embd * max(0, num("shortconv.l_cache") - 1) * 4
    elif num("ssm.inner_size") and num("ssm.state_size"):
        inner, d_state = num("ssm.inner_size"), num("ssm.state_size")
        conv = max(0, num("ssm.conv_kernel") - 1) * (inner + 2 * num("ssm.group_count") * d_state)
        state = (conv + d_state * inner) * 4

    out = []
    for i in range(n):
        if mla_per_token:
            per_token = mla_per_token if heads_kv[i] else 0
        elif swa[i]:
            per_token = heads_kv[i] * (k_swa + v_swa) * 2
        else:
            per_token = heads_kv[i] * (k_len + v_len) * 2
        layer_state = 0
        if arch in PARALLEL_HYBRID and state is not None:
            layer_state = state
        elif recurrent[i]:
            if state is None:  # a recurrent layer we can't size: keep the attention estimate
                per_token = per_token or info.head_count * (k_len + v_len) * 2
                out.append(KVLayer(per_token))
                continue
            per_token, layer_state = 0, state
        out.append(KVLayer(per_token, cells if swa[i] and per_token else 0, layer_state))
    return out


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
        nextn_layers=arch_key("nextn_predict_layers"),
        metadata={k: v for k, v in meta.items() if not isinstance(v, list) and not k.startswith("tokenizer.")},
    )

    # The header is untrusted: these feed allocations and the planner's arithmetic.
    if not 0 <= info.block_count <= min(MAX_BLOCKS, max(len(tensors), 1)):
        raise GGUFError(f"implausible block_count {info.block_count}")
    for field_name in (
        "context_length",
        "embedding_length",
        "head_count",
        "head_count_kv",
        "key_length",
        "value_length",
    ):
        if getattr(info, field_name) > MAX_DIMENSION:
            raise GGUFError(f"implausible {field_name} {getattr(info, field_name)}")
    if info.nextn_layers > info.block_count:
        raise GGUFError(f"implausible nextn_predict_layers {info.nextn_layers}")
    first_nextn = info.block_count - info.nextn_layers
    info.kv_layers = _kv_layout(meta, info)

    ordered = sorted(tensors, key=lambda t: t[1])
    layers = [0] * info.block_count
    token_embd = output = 0
    for i, (name, offset) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else data_size
        nbytes = end - offset
        if offset > data_size or nbytes < 0:
            raise GGUFError("GGUF tensor offsets are out of range")
        part = next((p for prefix, p in FOREIGN_PREFIXES.items() if name.startswith(prefix)), None)
        nextn = _NEXTN.match(name)
        if nextn and not first_nextn <= int(nextn.group(1)) < info.block_count:
            part = "MTP"  # NextN tensors in a block the header doesn't declare as a NextN layer
        if part is not None:
            info.foreign[part] = info.foreign.get(part, 0) + 1
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
