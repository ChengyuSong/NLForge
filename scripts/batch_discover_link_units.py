#!/usr/bin/env python3
"""Batch discover-link-units script for processing multiple projects.

Iterates over projects and runs discover-link-units for each, writing
func-scans/<project>/link_units.json. Projects that already have
link_units.json are skipped unless --force is given.

For CMake+Ninja builds, discovery is deterministic (no LLM needed).
For autotools/make builds, an LLM agent is used.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpr_utils import find_project_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"


def run_discover_link_units(
    project_name: str,
    project_path: Path,
    build_dir: Path,
    backend: str,
    model: str | None,
    llm_host: str | None,
    llm_port: int | None,
    verbose: bool,
) -> tuple[bool, str, float]:
    """Run discover-link-units for a single project.

    Returns (success, error_message, duration_seconds).
    """
    cmd = [
        "llm-summary",
        "discover-link-units",
        "--project-path", str(project_path),
        "--build-dir", str(build_dir),
        "--backend", backend,
    ]

    if model:
        cmd.extend(["--model", model])
    if llm_host:
        cmd.extend(["--llm-host", llm_host])
    if llm_port:
        cmd.extend(["--llm-port", str(llm_port)])
    if verbose:
        cmd.append("--verbose")

    start = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=str(REPO_ROOT),
        )
        duration = (datetime.now() - start).total_seconds()

        if result.returncode == 0:
            return True, "", duration
        else:
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            error_msg = (stderr + stdout).strip()
            return False, error_msg, duration

    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        return False, str(e), duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch discover link units for multiple projects"
    )
    parser.add_argument(
        "--projects-json",
        type=Path,
        default=Path("scripts/gpr_projects.json"),
        help="Path to gpr_projects.json file",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/data/csong/opensource"),
        help="Root directory where projects are cloned",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/data/csong/build-artifacts"),
        help="Root directory for build outputs",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="claude",
        choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        help="LLM backend (only used for non-Ninja/non-deterministic builds)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (optional, uses backend default if not specified)",
    )
    parser.add_argument(
        "--llm-host",
        type=str,
        help="Host for local LLM backends (ollama, llamacpp)",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        help="Port for local LLM backends",
    )
    parser.add_argument(
        "--tier",
        type=int,
        help="Only process projects from this tier (1 or 2)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only process projects whose name contains this string (case-insensitive)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of projects to process",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N projects",
    )
    parser.add_argument(
        "--skip-list",
        type=Path,
        help="File with project names to skip (one per line)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if link_units.json already exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output JSON report file (optional)",
    )

    args = parser.parse_args()

    if not args.projects_json.exists():
        print(f"Error: Projects JSON not found: {args.projects_json}")
        sys.exit(1)

    if not args.source_dir.exists():
        print(f"Error: Source directory not found: {args.source_dir}")
        sys.exit(1)

    # Load skip list
    skip_set: set[str] = set()
    if args.skip_list:
        if not args.skip_list.exists():
            print(f"Error: Skip list not found: {args.skip_list}")
            sys.exit(1)
        with open(args.skip_list) as f:
            skip_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(skip_set)} projects to skip from {args.skip_list}")

    # Load and filter projects
    with open(args.projects_json) as f:
        projects = json.load(f)

    print(f"Loaded {len(projects)} projects from {args.projects_json}")

    if args.tier:
        projects = [p for p in projects if p.get("tier") == args.tier]
        print(f"Filtered to {len(projects)} projects in tier {args.tier}")

    if args.filter:
        filter_str = args.filter.lower()
        projects = [p for p in projects if filter_str in p["name"].lower()]
        print(f"Filtered to {len(projects)} projects matching '{args.filter}'")

    if args.skip > 0:
        projects = projects[args.skip:]
        print(f"Skipped first {args.skip} projects, {len(projects)} remaining")

    if args.limit:
        projects = projects[: args.limit]
        print(f"Limited to {args.limit} projects")

    results = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "model": args.model,
        "total": len(projects),
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "projects": [],
    }

    for i, project in enumerate(projects, 1):
        project_name = project["name"]
        print(f"\n[{i}/{len(projects)}] {project_name}")

        project_path = find_project_dir(project, args.source_dir)

        if project_path and project_path.name in skip_set:
            print(f"  ⏭️  Skipping (in skip list)")
            results["skipped"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "skipped",
                "reason": "in_skip_list",
            })
            continue

        if not project_path:
            print(f"  ⚠️  Project directory not found — skipping")
            results["skipped"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "skipped",
                "reason": "directory_not_found",
            })
            continue

        build_dir = args.build_root / project_path.name
        if not build_dir.exists():
            print(f"  ⚠️  Build dir not found ({build_dir}) — skipping (run build-learn first)")
            results["skipped"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "skipped",
                "reason": "build_dir_not_found",
                "path": str(project_path),
            })
            continue

        # Skip if already discovered
        link_units_path = FUNC_SCANS_DIR / project_path.name / "link_units.json"
        if link_units_path.exists() and not args.force:
            print(f"  ✅ Already discovered ({link_units_path}) — skipping")
            results["skipped"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "skipped",
                "reason": "already_exists",
                "path": str(project_path),
                "link_units_path": str(link_units_path),
            })
            continue

        success, error_msg, duration = run_discover_link_units(
            project_name=project_name,
            project_path=project_path,
            build_dir=build_dir,
            backend=args.backend,
            model=args.model,
            llm_host=args.llm_host,
            llm_port=args.llm_port,
            verbose=args.verbose,
        )

        if success:
            # Count link units discovered
            n_units = 0
            if link_units_path.exists():
                try:
                    with open(link_units_path) as f:
                        lu_data = json.load(f)
                    n_units = len(lu_data.get("link_units", lu_data.get("targets", [])))
                except Exception:
                    pass
            print(f"  ✅ Success! {n_units} link unit(s) discovered ({duration:.1f}s)")
            results["succeeded"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "success",
                "duration": duration,
                "link_units": n_units,
                "path": str(project_path),
                "build_dir": str(build_dir),
            })
        else:
            print(f"  ❌ Failed ({duration:.1f}s)")
            if error_msg:
                print(f"  Error: {error_msg[:200]}")
            results["failed"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "failed",
                "duration": duration,
                "error": error_msg,
                "path": str(project_path),
                "build_dir": str(build_dir),
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {results['succeeded']} succeeded, "
          f"{results['failed']} failed, {results['skipped']} skipped "
          f"(total {results['total']})")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report written to {args.report}")

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
