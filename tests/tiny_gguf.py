"""Write a tiny but real llama-architecture GGUF model (random weights) for tests.

Pure Python, no numpy: a header, a ~300-token SentencePiece-style vocabulary
with byte fallback, and F32 tensors. llama.cpp loads and runs it (the output
is gibberish, which is all the end-to-end tests need).
"""

from __future__ import annotations

import random
import struct
from pathlib import Path
from typing import Any

ALIGN = 32
U32, I32, F32, BOOL, STRING, ARRAY, U64 = 4, 5, 6, 7, 8, 9, 10
_ITEM = {F32: "<f", I32: "<i", U32: "<I", BOOL: "<?"}


def _str(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value: Any) -> bytes:
    out = _str(key)
    if isinstance(value, bool):
        return out + struct.pack("<I?", BOOL, value)
    if isinstance(value, str):
        return out + struct.pack("<I", STRING) + _str(value)
    if isinstance(value, float):
        return out + struct.pack("<If", F32, value)
    if isinstance(value, int):
        if value >= 2**32:
            return out + struct.pack("<IQ", U64, value)
        return out + struct.pack("<II", U32, value)
    if isinstance(value, tuple):  # (item type, items)
        kind, items = value
        body = b"".join(_str(i) if kind == STRING else struct.pack(_ITEM[kind], i) for i in items)
        return out + struct.pack("<IIQ", ARRAY, kind, len(items)) + body
    raise TypeError(type(value))


def vocab() -> tuple[list[str], list[float], list[int]]:
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    types = [2, 3, 3] + [6] * 256
    words = ["▁" + w for w in "the a of and to in is you that it he was for on are as with".split()]
    tokens += words + list("abcdefghijklmnopqrstuvwxyz")
    types += [1] * (len(words) + 26)
    scores = [0.0] * 3 + [-1000.0] * 256 + [-float(i) for i in range(len(tokens) - 259)]
    return tokens, scores, types


def write_tiny_llama(
    path: Path,
    n_layer: int = 4,
    n_embd: int = 64,
    n_head: int = 4,
    n_head_kv: int = 2,
    n_ff: int = 128,
    ctx: int = 512,
    tied: bool = False,
    seed: int = 1,
    extra_kv: dict[str, Any] | None = None,
    arch: str = "llama",
    extra_tensors: list[tuple[str, list[int]]] | None = None,
    n_expert: int = 0,
    n_expert_used: int = 2,
) -> Path:
    """A llama-shaped model; ``arch`` renames the architecture (and its keys), ``extra_tensors`` adds tensors."""
    tokens, scores, types = vocab()
    n_vocab = len(tokens)
    n_kv_embd = n_embd // n_head * n_head_kv
    kv = {
        "general.architecture": arch,
        "general.name": "tiny",
        "general.alignment": ALIGN,
        "general.file_type": 0,
        f"{arch}.context_length": ctx,
        f"{arch}.embedding_length": n_embd,
        f"{arch}.block_count": n_layer,
        f"{arch}.feed_forward_length": n_ff,
        f"{arch}.attention.head_count": n_head,
        f"{arch}.attention.head_count_kv": n_head_kv,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{arch}.rope.dimension_count": n_embd // n_head,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": (STRING, tokens),
        "tokenizer.ggml.scores": (F32, scores),
        "tokenizer.ggml.token_type": (I32, types),
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.unknown_token_id": 0,
        **({"llama.expert_count": n_expert, "llama.expert_used_count": n_expert_used} if n_expert else {}),
        **(extra_kv or {}),
    }
    tensors: list[tuple[str, list[int]]] = [("token_embd.weight", [n_embd, n_vocab])]
    for i in range(n_layer):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", [n_embd]),
            (p + "attn_q.weight", [n_embd, n_embd]),
            (p + "attn_k.weight", [n_embd, n_kv_embd]),
            (p + "attn_v.weight", [n_embd, n_kv_embd]),
            (p + "attn_output.weight", [n_embd, n_embd]),
            (p + "ffn_norm.weight", [n_embd]),
        ]
        if n_expert:  # Mixtral-style MoE: a router and stacked experts
            tensors += [
                (p + "ffn_gate_inp.weight", [n_embd, n_expert]),
                (p + "ffn_gate_exps.weight", [n_embd, n_ff, n_expert]),
                (p + "ffn_down_exps.weight", [n_ff, n_embd, n_expert]),
                (p + "ffn_up_exps.weight", [n_embd, n_ff, n_expert]),
            ]
        else:
            tensors += [
                (p + "ffn_gate.weight", [n_embd, n_ff]),
                (p + "ffn_down.weight", [n_ff, n_embd]),
                (p + "ffn_up.weight", [n_embd, n_ff]),
            ]
    tensors.append(("output_norm.weight", [n_embd]))
    if not tied:
        tensors.append(("output.weight", [n_embd, n_vocab]))
    tensors += extra_tensors or []

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    header += b"".join(_kv(k, v) for k, v in kv.items())
    rng = random.Random(seed)
    infos, blobs, offset = b"", [], 0
    for name, dims in tensors:
        count = 1
        for d in dims:
            count *= d
        if name.endswith("norm.weight"):
            data = struct.pack(f"<{count}f", *([1.0] * count))
        else:
            data = struct.pack(f"<{count}f", *(rng.uniform(-0.05, 0.05) for _ in range(count)))
        infos += _str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", 0, offset)  # type F32, offset
        pad = -len(data) % ALIGN
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    head = header + infos
    head += b"\0" * (-len(head) % ALIGN)
    path.write_bytes(head + b"".join(blobs))
    return path
