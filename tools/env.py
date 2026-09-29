"""Read secrets from .env without ever printing them.

The file is gitignored and is the only place keys live. Nothing in this module
returns a key to stdout; `describe()` reports presence and length only, which is
enough to debug a bad paste without the value entering a log, a terminal
scrollback, or a transcript.

Accepts the loose formats people actually write:
    name = 'value'      name="value"      export NAME=value      NAME=value
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Accept the several names the same secret gets written under.
ALIASES = {
    "GROQ_API_KEY":   ("groq_key", "groq_api_key", "GROQ_KEY", "GROQ_API_KEY"),
    "COHERE_API_KEY": ("cohere_trail_key", "cohere_trial_key", "cohere_key",
                       "cohere_api_key", "COHERE_KEY", "COHERE_API_KEY"),
}

_LINE = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")


def _parse(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0] if raw.lstrip().startswith("#") else raw
        # allow several assignments separated by ';'
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
