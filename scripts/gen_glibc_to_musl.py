"""Generate src/llm_summary/data/libc_aliases/glibc_to_musl.json.

For every name in the bundled glibc abilist that is missing from the musl
scan DB, try to map it to a musl source name via:

  1. ``weak_alias(<priv>, <pub>)`` declarations scraped from musl source.
     Maps the glibc public name to the musl private/internal name that
     actually carries the implementation.

  2. ``__isoc23_<X>`` -> ``<X>`` C23 variants of strto*/scanf families.
     The C23 versions only differ in base-prefix handling for strto*;
     reusing the contract from the base routine is acceptable.

Names that resolve to neither path are *not* written to the JSON — they
remain cache misses, which forces a deliberate decision (add a manual
alias or a builtin contract) when an unmapped name is actually referenced
by a real project.

Run from the repo root:

    python scripts/gen_glibc_to_musl.py \
        --musl-source /data/csong/opensource/musl/src \
        --musl-db func-scans/musl/c/functions.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path

WA_RE = re.compile(r"weak_alias\(\s*(\w+)\s*,\s*(\w+)\s*\)")


def scan_weak_aliases(musl_src: Path) -> dict[str, str]:
    """Return {public_name: private_name} from weak_alias() declarations.

    First occurrence wins; later duplicates (rare in musl) are ignored.
    """
    out: dict[str, str] = {}
    for c in musl_src.rglob("*.c"):
        try:
            text = c.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in WA_RE.finditer(text):
            priv, pub = m.group(1), m.group(2)
            out.setdefault(pub, priv)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musl-source", type=Path, required=True,
                    help="Path to musl/src checkout")
    ap.add_argument("--musl-db", type=Path, required=True,
                    help="Path to func-scans/musl/c/functions.db")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "src/llm_summary/data/libc_aliases/glibc_to_musl.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from llm_summary.stdlib_cache import load_known_externals

    known = load_known_externals()
    con = sqlite3.connect(str(args.musl_db))
    musl_funcs = {r[0] for r in con.execute("SELECT name FROM functions")}
    diff = known - musl_funcs

    weak_alias_map = scan_weak_aliases(args.musl_source)

    mapping: dict[str, dict[str, str | None]] = {}

    # 1. Weak-alias resolutions
    for name in sorted(diff):
        priv = weak_alias_map.get(name)
        if priv and priv in musl_funcs:
            mapping[name] = {"musl": priv, "via": "weak_alias"}

    # 2. C23 fallback (only for names not already mapped)
    for name in sorted(diff):
        if name in mapping:
            continue
        if name.startswith("__isoc23_"):
            bare = name[len("__isoc23_"):]
            if bare in musl_funcs:
                mapping[name] = {"musl": bare, "via": "c23-variant"}

    # Stable ordering
    ordered = OrderedDict(sorted(mapping.items()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")

    print(f"abilist names:        {len(known):>5}")
    print(f"musl scan names:      {len(musl_funcs):>5}")
    print(f"diff (need mapping):  {len(diff):>5}")
    print(f"resolved (written):   {len(ordered):>5}")
    via_wa = sum(1 for v in ordered.values() if v["via"] == "weak_alias")
    via_c23 = sum(1 for v in ordered.values() if v["via"] == "c23-variant")
    print(f"  via weak_alias:     {via_wa:>5}")
    print(f"  via c23-variant:    {via_c23:>5}")
    print(f"unresolved (skipped): {len(diff) - len(ordered):>5}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
