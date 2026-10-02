"""Ask the local Ollama for its version and loaded models (adapted from OllamaGate).

A loaded model whose ``size_vram`` is below its ``size`` is partly on the CPU:
Ollama *spilled* layers to system RAM, which is slow. That is exactly the
case where a helper GPU can help, so the hub surfaces it as a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from ..textsafe import safe_text

PROBE_TIMEOUT = 3.0


@dataclass
class LoadedModel:
    name: str
    size: int
    size_vram: int
    digest: str = ""

    @property
    def spilled(self) -> bool:
        return self.size > 0 and self.size_vram < self.size

    @property
    def vram_percent(self) -> int:
        return 100 if self.size <= 0 else min(100, self.size_vram * 100 // self.size)


@dataclass
class OllamaStatus:
    reachable: bool
    version: str = ""
    models: list[LoadedModel] = field(default_factory=list)
    error: str = ""


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**63 else 0


def parse_ps(data: object) -> list[LoadedModel]:
    """``/api/ps`` JSON -> models; anything malformed is skipped."""
    raw = data.get("models") if isinstance(data, dict) else None
    models = []
    for m in raw if isinstance(raw, list) else []:
        if isinstance(m, dict):
            models.append(
                LoadedModel(
                    name=safe_text(str(m.get("name") or m.get("model") or "?"), 120),
                    size=_int(m.get("size")),
                    size_vram=_int(m.get("size_vram")),
                    digest=safe_text(str(m.get("digest", "")), 80),
                )
            )
    return models


async def probe(client: httpx.AsyncClient) -> OllamaStatus:
    try:
        version = (await client.get("/api/version", timeout=PROBE_TIMEOUT)).json().get("version", "")
        ps = (await client.get("/api/ps", timeout=PROBE_TIMEOUT)).json()
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return OllamaStatus(reachable=False, error=type(exc).__name__)
    return OllamaStatus(reachable=True, version=safe_text(str(version), 40), models=parse_ps(ps))


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"
