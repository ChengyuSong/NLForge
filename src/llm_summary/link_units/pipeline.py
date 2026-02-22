"""Shared helpers for link-unit-aware batch pipeline scripts.

Provides:
  load_link_units(path)           -- load link_units.json, return list of dicts
  topo_sort_link_units(units)     -- DFS topological sort (deps before dependents)
  build_output_index(units)       -- map output filename/basename -> link unit
  resolve_dep_db_paths(lu, idx, project_scan_dir)
                                  -- get dep DB Paths for a target
  update_link_units_file(path, units)
                                  -- write updated link_units.json back to disk
"""

from __future__ import annotations

import json
from pathlib import Path


def load_link_units(link_units_path: Path) -> tuple[dict, list[dict]]:
    """Load link_units.json.

    Returns (raw_data, link_units_list).
    raw_data is the full JSON object (for write-back).
    link_units_list is the array of link unit dicts.
    """
    with open(link_units_path) as f:
        data = json.load(f)
    units = data.get("link_units", data.get("targets", []))
    return data, units


def build_output_index(link_units: list[dict]) -> dict[str, dict]:
    """Map output filename (full and basename) to link unit dict."""
    idx: dict[str, dict] = {}
    for lu in link_units:
        idx[lu["output"]] = lu
        idx[Path(lu["output"]).name] = lu
    return idx


def topo_sort_link_units(link_units: list[dict]) -> list[dict]:
    """Return link units in dependency order (deps before dependents).

    Uses iterative DFS to avoid recursion limits on large graphs.
    Raises ValueError on cycles.
    """
    by_output = build_output_index(link_units)
    by_name = {lu["name"]: lu for lu in link_units}

    visited: set[str] = set()
    in_stack: set[str] = set()
    order: list[dict] = []

    def visit(lu: dict) -> None:
        name = lu["name"]
        if name in visited:
            return
        if name in in_stack:
            raise ValueError(f"Cycle detected involving link unit '{name}'")
        in_stack.add(name)
        for dep_output in lu.get("link_deps", []):
            dep_lu = by_output.get(dep_output)
            if dep_lu:
                visit(dep_lu)
        in_stack.discard(name)
        visited.add(name)
        order.append(lu)

    for lu in link_units:
        visit(lu)

    return order


def resolve_dep_db_paths(
    lu: dict,
    output_index: dict[str, dict],
    project_scan_dir: Path,
) -> list[Path]:
    """Return existing DB paths for the direct link_deps of a link unit.

    Prefers lu["db_path"] written by batch_call_graph_gen; falls back to
    func-scans/<project>/<target>/functions.db.
    """
    dep_paths: list[Path] = []
    for dep_output in lu.get("link_deps", []):
        dep_lu = output_index.get(dep_output)
        if not dep_lu:
            continue
        db_str = dep_lu.get("db_path")
        db_path = Path(db_str) if db_str else project_scan_dir / dep_lu["name"] / "functions.db"
        if db_path.exists():
            dep_paths.append(db_path)
    return dep_paths


def update_link_units_file(link_units_path: Path, data: dict) -> None:
    """Write the (possibly mutated) link_units.json data back to disk."""
    with open(link_units_path, "w") as f:
        json.dump(data, f, indent=2)
