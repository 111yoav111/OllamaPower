"""Find the GGUF file Ollama stores for a model, so llama.cpp can load the same weights.

Split mode reuses Ollama's download: no second copy of the model. Ollama
says where the weights are in ``/api/show``: its generated Modelfile starts
with ``FROM <models dir>/blobs/sha256-<digest>`` (later ``FROM`` lines are
vision projectors). That path is authoritative even when Ollama runs as a
service with its own ``OLLAMA_MODELS``. The manifest next to the blobs is read
too when possible, to catch what the Modelfile doesn't show (sharded weights).

Split mode can't use a model whose weights aren't one GGUF file (sharded
models, safetensors imports), has adapters, isn't a text model, has an
architecture the pinned llama.cpp doesn't know, or keeps vision/audio/MTP
tensors in Ollama's own layout inside the GGUF; ``resolve`` says so in
owner-readable words, before anything is started.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..llamacpp.archs import architecture_problem
from ..textsafe import safe_text
from .client import PROBE_HEADERS
from .gguf import GGUFError, GGUFInfo, read_gguf

SHOW_TIMEOUT = 30.0
MEDIA_MODEL = "application/vnd.ollama.image.model"
MEDIA_PROJECTOR = "application/vnd.ollama.image.projector"
MEDIA_ADAPTER = "application/vnd.ollama.image.adapter"
_BLOB = re.compile(r"[/\\]blobs[/\\]sha256[-:]([0-9a-f]{64})$")
MAX_MANIFEST = 1024 * 1024


class ModelUnavailable(Exception):
    """This model can't be used in split mode (the message says why)."""


@dataclass
class ModelFiles:
    name: str
    path: Path  # the GGUF weights
    digest: str
    projectors: list[Path] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    gguf: GGUFInfo | None = None

    @property
    def vision(self) -> bool:
        return bool(self.projectors) or "vision" in self.capabilities


def parse_name(name: str) -> tuple[str, str, str, str]:
    """``[host/][namespace/]model[:tag]`` -> its four parts, with Ollama's defaults."""
    head, _, last = name.rpartition("/")
    tag = "latest"
    if ":" in last:
        last, tag = last.split(":", 1)
    parts = [*(head.split("/") if head else []), last]
    host, namespace = "registry.ollama.ai", "library"
    if len(parts) == 3:
        host, namespace, model = parts
    elif len(parts) == 2:
        namespace, model = parts
    else:
        model = parts[0]
    return host, namespace, model, tag


def _from_lines(modelfile: str) -> list[str]:
    out = []
    for line in modelfile.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            out.append(stripped[5:].strip().strip('"'))
    return out


def _blob(path: str) -> str | None:
    match = _BLOB.search(path)
    return match.group(1) if match else None


def read_manifest(models_dir: Path, name: str) -> dict[str, Any] | None:
    host, namespace, model, tag = parse_name(name)
    path = models_dir / "manifests" / host / namespace / model / tag
    try:
        if path.stat().st_size > MAX_MANIFEST:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _check_manifest(manifest: dict[str, Any], name: str) -> list[str]:
    """Projector digests; refuses what split mode can't load."""
    layers = [x for x in manifest.get("layers", []) if isinstance(x, dict)]
    kinds = [x.get("mediaType") for x in layers]
    if kinds.count(MEDIA_MODEL) > 1:
        raise ModelUnavailable(f"{name} is stored in several files (sharded); split mode needs a single GGUF")
    if MEDIA_ADAPTER in kinds:
        raise ModelUnavailable(f"{name} uses an adapter (LoRA); split mode can't apply it")
    return [str(x.get("digest", "")).split(":")[-1] for x in layers if x.get("mediaType") == MEDIA_PROJECTOR]


def default_models_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("OLLAMA_MODELS"):
        dirs.append(Path(os.environ["OLLAMA_MODELS"]))
    dirs.append(Path.home() / ".ollama" / "models")
    dirs.append(Path("/usr/share/ollama/.ollama/models"))  # the Linux service install
    return dirs


async def show(client: httpx.AsyncClient, name: str) -> dict[str, Any]:
    try:
        r = await client.post("/api/show", json={"model": name}, timeout=SHOW_TIMEOUT, headers=PROBE_HEADERS)
    except httpx.HTTPError as exc:
        raise ModelUnavailable(f"Ollama is not reachable ({type(exc).__name__})") from None
    if r.status_code == 404:
        raise ModelUnavailable(f"{name} is not installed in Ollama (ollama pull {name})")
    if r.status_code != 200:
        raise ModelUnavailable(f"Ollama could not describe {name} (HTTP {r.status_code})")
    try:
        data = r.json()
    except ValueError:
        raise ModelUnavailable(f"Ollama sent an unreadable answer for {name}") from None
    if not isinstance(data, dict):
        raise ModelUnavailable(f"Ollama sent an unreadable answer for {name}")
    return data


def locate(data: dict[str, Any], name: str, models_dirs: list[Path] | None = None) -> ModelFiles:
    """Turn an ``/api/show`` answer into the files on disk (blocking: reads the disk)."""
    capabilities = [c for c in data.get("capabilities") or [] if isinstance(c, str)]
    if capabilities and "completion" not in capabilities:
        raise ModelUnavailable(f"{name} is not a text-generation model ({', '.join(capabilities)})")
    details = data.get("details") if isinstance(data.get("details"), dict) else {}
    if details.get("format") not in (None, "", "gguf"):
        raise ModelUnavailable(f"{name} is in {details.get('format')} format; split mode needs GGUF")
    if data.get("remote_host"):
        raise ModelUnavailable(f"{name} is a cloud model; split mode runs local models only")

    froms = [p for p in _from_lines(str(data.get("modelfile", ""))) if _blob(p)]
    model_path: Path | None = None
    projectors: list[Path] = []
    manifest = None
    if froms:
        model_path = Path(froms[0])
        projectors = [Path(p) for p in froms[1:]]
        manifest = read_manifest(model_path.parent.parent, name)
    else:
        # Older or unusual Ollama: look for the manifest in the usual places.
        for base in models_dirs or default_models_dirs():
            manifest = read_manifest(base, name)
            if manifest is None:
                continue
            layers = [x for x in manifest.get("layers", []) if isinstance(x, dict)]
            digest = next((str(x.get("digest", "")) for x in layers if x.get("mediaType") == MEDIA_MODEL), "")
            if digest:
                model_path = base / "blobs" / digest.replace(":", "-")
                break
    if model_path is None:
        raise ModelUnavailable(f"could not find where Ollama stores {name}")
    if manifest is not None:
        for digest in _check_manifest(manifest, name):
            candidate = model_path.parent / f"sha256-{digest}"
            if candidate not in projectors:
                projectors.append(candidate)

    if not os.access(model_path, os.R_OK):
        raise ModelUnavailable(
            f"cannot read {model_path}: Ollama's model files must be readable by the user running the hub "
            "(see `ollamapower doctor`)"
        )
    try:
        info = read_gguf(model_path)
    except (GGUFError, OSError) as exc:
        raise ModelUnavailable(f"{name}: {exc}") from None
    if info.split_count > 1:
        raise ModelUnavailable(f"{name} is stored in several files (sharded); split mode needs a single GGUF")
    if info.block_count <= 0:
        raise ModelUnavailable(f"{name} has no layers split mode can place ({safe_text(info.architecture, 60)})")
    check_llamacpp_can_load(info, name)
    return ModelFiles(
        name=name,
        path=model_path,
        digest=_blob(str(model_path)) or "",
        projectors=projectors,
        capabilities=capabilities,
        gguf=info,
    )


def check_llamacpp_can_load(info: GGUFInfo, name: str) -> None:
    """Refuse, before anything starts, a GGUF the pinned llama.cpp can't serve as a text model.

    Without this the failure comes from llama-server itself, after Ollama has
    unloaded its models and the helpers have started rpc-server.
    """
    problem = architecture_problem(info.architecture)
    if problem:
        raise ModelUnavailable(f"{name} can't run in split mode: {problem}; Ollama serves it")
    if info.foreign:
        parts = " and ".join(sorted(info.foreign))
        raise ModelUnavailable(
            f"{name} can't run in split mode: its GGUF holds {parts} tensors in Ollama's own layout, "
            "which llama.cpp doesn't load; Ollama serves it"
        )


async def resolve(client: httpx.AsyncClient, name: str, models_dirs: list[Path] | None = None) -> ModelFiles:
    """Where ``name``'s weights are and what's in them."""
    data = await show(client, name)
    return await asyncio.to_thread(locate, data, name, models_dirs)
