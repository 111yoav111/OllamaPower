"""Pin a llama.cpp release: download every archive OllamaPower may use and record its SHA-256.

llama.cpp publishes no checksums, so we compute them once, here, and ship
them in ``ollamapower/llamacpp/pins.json``. Every machine then refuses any
download that doesn't match. Run it from a trusted network:

    python scripts/pin_llamacpp.py b11347
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollamapower.llamacpp.install import RELEASE_URL  # noqa: E402
from ollamapower.llamacpp.variants import archives_for  # noqa: E402

TARGETS = [
    ("macos", "arm64", "metal"),
    ("macos", "x64", "cpu"),
    ("linux", "x64", "cpu"),
    ("linux", "arm64", "cpu"),
    ("linux", "x64", "vulkan"),
    ("linux", "arm64", "vulkan"),
    ("linux", "x64", "cuda12"),
    ("linux", "x64", "cuda13"),
    ("linux", "x64", "rocm"),
    ("windows", "x64", "cpu"),
    ("windows", "arm64", "cpu"),
    ("windows", "x64", "vulkan"),
    ("windows", "x64", "cuda12"),
    ("windows", "x64", "cuda13"),
    ("windows", "x64", "rocm"),
]


def main(tag: str) -> None:
    names = sorted({name for t in TARGETS for name in archives_for(tag, *t).archives})
    archives = {}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client:
        for name in names:
            h, size = hashlib.sha256(), 0
            with client.stream("GET", RELEASE_URL.format(tag=tag, name=name)) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(1024 * 1024):
                    h.update(chunk)
                    size += len(chunk)
            archives[name] = {"size": size, "sha256": h.hexdigest()}
            print(f"{size / 1e6:8.1f} MB  {h.hexdigest()}  {name}", file=sys.stderr)
    out = Path(__file__).resolve().parent.parent / "ollamapower" / "llamacpp" / "pins.json"
    out.write_text(json.dumps({"tag": tag, "build": int(tag.lstrip("b")), "archives": archives}, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
