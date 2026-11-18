"""Read secrets from .env without ever printing them."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

ALIASES = {
    "GROQ_API_KEY":   ("groq_key", "groq_api_key", "GROQ_KEY", "GROQ_API_KEY"),
    "COHERE_API_KEY": ("cohere_trail_key", "cohere_trial_key", "cohere_key",
                       "cohere_api_key", "COHERE_KEY", "COHERE_API_KEY"),
    "NVIDIA_API_KEY": ("nvidia_key", "nvidia_api_key", "NVIDIA_KEY", "NVIDIA_API_KEY"),
    "OPENROUTER_API_KEY": ("openrouter_key", "openrouter_api_key", "OPENROUTER_KEY",
                           "OPENROUTER_API_KEY"),
}

_LINE = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")


def _parse(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0] if raw.lstrip().startswith("#") else raw
        for part in line.split(";"):
            m = _LINE.match(part)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            if v:
                out[k] = v
    return out


def load(path: Path = ENV_FILE) -> None:
    """Populate os.environ with canonical names. Existing env wins."""
    raw = _parse(path)
    for canonical, names in ALIASES.items():
        if os.environ.get(canonical):
            continue
        for n in names:
            if raw.get(n):
                os.environ[canonical] = raw[n]
                break


def get(name: str) -> Optional[str]:
    load()
    return os.environ.get(name)


def describe() -> Dict[str, str]:
    """Presence and shape only -- never the value."""
    load()
    out = {}
    for canonical in ALIASES:
        v = os.environ.get(canonical)
        out[canonical] = f"set ({len(v)} chars, ends …{v[-3:]})" if v else "MISSING or empty"
    return out


if __name__ == "__main__":
    for k, v in describe().items():
        print(f"  {k:<18} {v}")
