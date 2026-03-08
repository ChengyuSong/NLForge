"""Deterministic parsing skills for link-unit discovery.

Pure Python functions (no LLM) that parse build system artifacts
to identify link units (libraries and executables) and their
constituent object/.bc files.
"""

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Docker container path translation
# /workspace/src  -> project source dir
# /workspace/build -> build dir
# ---------------------------------------------------------------------------

def _is_docker_path(path: str) -> bool:
    return path.startswith("/workspace/")


def _resolve_host_path(container_path: str, project_source_dir: Path, build_dir: Path) -> Path:
    if not _is_docker_path(container_path):
        return Path(container_path)
    remainder = container_path[len("/workspace/"):]
    if remainder.startswith("src/"):
        return project_source_dir / remainder[len("src/"):]
    elif remainder == "src":
        return project_source_dir
    elif remainder.startswith("build/"):
        return build_dir / remainder[len("build/"):]
    elif remainder == "build":
        return build_dir
    else:
        return build_dir / remainder


def _translate_arg(arg: str, project_source_dir: Path, build_dir: Path) -> str:
    if _is_docker_path(arg):
        return str(_resolve_host_path(arg, project_source_dir, build_dir))
    for prefix in ("-I", "-isystem", "-isysroot", "-include", "-iprefix",
                   "-iwithprefix", "-iwithprefixbefore", "-iquote"):
        if arg.startswith(prefix) and _is_docker_path(arg[len(prefix):]):
            translated = _resolve_host_path(arg[len(prefix):], project_source_dir, build_dir)
            return f"{prefix}{translated}"
    return arg


# Regex for CMake/Ninja link rules
# Matches lines like:
#   build libz.a: C_STATIC_LIBRARY_LINKER__zlibstatic_ obj1.o obj2.o | dep1 || order_dep
#   build bin/exe: CXX_EXECUTABLE_LINKER__myexe_ obj.o | libfoo.a || libfoo.a
_LINK_RULE_RE = re.compile(
    r"^build\s+(?P<output>\S+):\s+"
    r"(?P<rule>(?:C|CXX)_(?:STATIC_LIBRARY|SHARED_LIBRARY|EXECUTABLE)_LINKER__\S+)\s+"
    r"(?P<rest>.*)$"
)

# Map rule fragments to link-unit types
_RULE_TYPE_MAP = {
    "STATIC_LIBRARY": "static_library",
    "SHARED_LIBRARY": "shared_library",
    "EXECUTABLE": "executable",
}


def parse_ninja_targets(build_ninja: Path) -> dict[str, Any]:
    """Parse build.ninja for link rules and return structured link units.

    The Ninja format for CMake-generated link rules is:
        build <output>: <LANG>_<TYPE>_LINKER__<target>_ <objects> [| <deps>] [|| <order_deps>]

    Args:
        build_ninja: Path to build.ninja file

    Returns:
        Dict with 'targets' list, each containing:
        - name: target name (extracted from rule)
        - type: static_library / shared_library / executable
        - output: output file path (relative to build dir)
        - objects: list of object file paths
        - link_deps: list of link-time dependencies (from | section)
    """
    if not build_ninja.exists():
        return {"error": f"File not found: {build_ninja}", "targets": []}

    targets = []

    with open(build_ninja, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            m = _LINK_RULE_RE.match(line)
            if not m:
                continue

            output = m.group("output")
            rule = m.group("rule")
            rest = m.group("rest")

            # Determine type from rule name
            unit_type = "unknown"
            for fragment, mapped_type in _RULE_TYPE_MAP.items():
                if fragment in rule:
                    unit_type = mapped_type
                    break

            # Extract target name from rule: ..._LINKER__<name>_
            name_match = re.search(r"LINKER__(.+?)_$", rule)
            target_name = name_match.group(1) if name_match else Path(output).stem

            # Split rest into objects and dependencies
            # Format: obj1.o obj2.o | dep1.a dep2.so || order_dep1 order_dep2
            objects = []
            link_deps = []

            # Split on || first (order-only deps, we ignore these)
            parts = rest.split("||")
            main_part = parts[0].strip()

            # Split main part on | (implicit deps = link deps)
            dep_parts = main_part.split("|")
            obj_part = dep_parts[0].strip()
            dep_part = dep_parts[1].strip() if len(dep_parts) > 1 else ""

            # Parse objects (skip empty strings)
            objects = [o for o in obj_part.split() if o]

            # Parse link dependencies - filter out system libraries
            if dep_part:
                for dep in dep_part.split():
                    dep = dep.strip()
                    if dep and not dep.startswith("/usr/") and not dep.startswith("/lib/"):
                        link_deps.append(dep)

            targets.append({
                "name": target_name,
                "type": unit_type,
                "output": output,
                "objects": objects,
                "link_deps": link_deps,
            })

    return {"targets": targets}


ASM_EXTENSIONS = {".s", ".S", ".asm"}


def map_objects_to_bc(
    objects: list[str],
    build_dir: Path,
    compile_commands: list[dict] | None = None,
) -> dict[str, Any]:
    """Map object file paths to corresponding .bc files.

    Uses the same tier-1/tier-2 strategy as batch_call_graph_gen.py:
    - Tier 1: Look for .bc next to .o (from -save-temps=obj)
    - Tier 2: If -flto in compile flags, .o itself is LLVM bitcode

    Assembly source files (.s/.S/.asm) never produce .bc; their resolved
    source paths are collected separately in the returned 'asm_sources' list.

    Args:
        objects: List of object file paths (relative to build_dir)
        build_dir: Path to the build directory
        compile_commands: Optional loaded compile_commands.json entries

    Returns:
        Dict with 'mappings' (obj -> bc path), 'asm_sources', and 'stats'
    """
    mappings: dict[str, str | None] = {}
    asm_sources: list[str] = []
    stats = {"total": len(objects), "tier1": 0, "tier2": 0, "not_found": 0}

    # Build a lookup from output path to compile_commands entry
    cc_by_output: dict[str, dict] = {}
    if compile_commands:
        for entry in compile_commands:
            output = entry.get("output")
            if output:
                directory = entry.get("directory", "")
                o_path = Path(output)
                if not o_path.is_absolute():
                    o_path = Path(directory) / o_path
                cc_by_output[str(o_path.resolve())] = entry

    for obj in objects:
        obj_path = Path(obj)
        if not obj_path.is_absolute():
            obj_path = build_dir / obj_path

        bc_path = None

        # Tier 1: Look for .bc next to .o
        # With -save-temps=obj, clang writes <stem>.bc next to <stem>.<ext>.o
        # E.g., png.c.o -> png.bc
        source_stem = _get_source_stem(obj_path)
        if source_stem:
            candidate = obj_path.parent / (source_stem + ".bc")
            if candidate.exists():
                bc_path = str(candidate.resolve())
                stats["tier1"] += 1

        # Tier 2: If -flto, the .o file IS bitcode
        if bc_path is None:
            resolved_obj = str(obj_path.resolve())
            cc_entry = cc_by_output.get(resolved_obj)
            if cc_entry and _has_flto(cc_entry) and obj_path.exists():
                bc_path = resolved_obj
                stats["tier2"] += 1

        if bc_path is None:
            # Check if this object came from an assembly source file
            resolved_obj = str(obj_path.resolve())
            cc_entry = cc_by_output.get(resolved_obj)
            if cc_entry:
                src = cc_entry.get("file", "")
                if src and Path(src).suffix in ASM_EXTENSIONS:
                    directory = cc_entry.get("directory", "")
                    src_path = Path(src) if Path(src).is_absolute() else Path(directory) / src
                    asm_sources.append(str(src_path.resolve()))
                    continue  # not counted as not_found
            stats["not_found"] += 1

        mappings[obj] = bc_path

    return {"mappings": mappings, "asm_sources": asm_sources, "stats": stats}


def _get_source_stem(obj_path: Path) -> str | None:
    """Extract the source file stem from an object path.

    CMake object files use patterns like:
      CMakeFiles/target.dir/foo.c.o -> stem is "foo"
      CMakeFiles/target.dir/bar.cpp.o -> stem is "bar"
    """
    name = obj_path.name  # e.g., "foo.c.o"
    # Remove .o suffix
    if name.endswith(".o"):
        without_o = name[:-2]  # "foo.c"
        # Remove source extension
        stem = Path(without_o).stem  # "foo"
        return stem
    return None


def _has_flto(entry: dict) -> bool:
    """Check if the compile command uses -flto."""
    if "arguments" in entry:
        return any("-flto" in a for a in entry["arguments"])
    cmd = entry.get("command", "")
    return "-flto" in cmd


def run_ar_t(archive: Path) -> list[str]:
    """Run `ar t <archive>` to list archive members.

    Args:
        archive: Path to the archive file (.a)

    Returns:
        List of member object file names
    """
    if not archive.exists():
        return []

    try:
        result = subprocess.run(
            ["ar", "t", str(archive)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def prescan_build_dir(build_dir: Path, verbose: bool = False) -> dict:
    """Pre-scan the build directory to find archives and ELF executables.

    Used to size the agent's turn budget and feed into heuristic discovery.

    Returns:
        Dict with 'archives', 'executables', and 'estimated_targets'.
    """
    build_dir = Path(build_dir).resolve()

    # Find all .a files
    archives = []
    for a in build_dir.rglob("*.a"):
        try:
            archives.append(str(a.relative_to(build_dir)))
        except ValueError:
            archives.append(str(a))

    # Find ELF executables on disk
    executables = []
    try:
        find_result = subprocess.run(
            ["find", str(build_dir), "-type", "f", "-executable"],
            capture_output=True, text=True, timeout=30,
        )
        candidates = [f.strip() for f in find_result.stdout.splitlines() if f.strip()]

        if candidates:
            # Batch-check with `file` to filter for ELF executables
            batch_size = 200
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                file_result = subprocess.run(
                    ["file", "--brief", *batch],
                    capture_output=True, text=True, timeout=30,
                )
                for path, desc in zip(batch, file_result.stdout.splitlines(), strict=False):
                    if "ELF" in desc and ("executable" in desc or "pie executable" in desc):
                        try:
                            rel = str(Path(path).relative_to(build_dir))
                        except ValueError:
                            rel = path
                        executables.append(rel)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        if verbose:
            print(f"[prescan] ELF scan failed: {e}")

    estimated_targets = len(archives) + len(executables)

    if verbose:
        print(f"[prescan] Found {len(archives)} archives, "
              f"{len(executables)} ELF executables, "
              f"~{estimated_targets} estimated link units")

    return {
        "archives": sorted(archives),
        "executables": sorted(executables),
        "estimated_targets": estimated_targets,
    }


def _parse_makefile_variables(makefile: Path) -> dict[str, str]:
    """Parse a Makefile and extract variable assignments.

    Handles simple `VAR = value` and `VAR = value \\\\` continuation lines.
    Does NOT handle recursive expansion — callers must resolve $(VAR) refs.

    Returns:
        Dict of variable name -> raw value string.
    """
    variables: dict[str, str] = {}
    if not makefile.exists():
        return variables

    current_var = None
    current_val = ""

    with open(makefile, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()

            # Continuation of previous line
            if current_var is not None:
                if line.endswith("\\"):
                    current_val += " " + line[:-1].strip()
                    continue
                else:
                    current_val += " " + line.strip()
                    variables[current_var] = current_val.strip()
                    current_var = None
                    current_val = ""
                    continue

            # New variable assignment (VAR = ... or VAR := ...)
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:+?]?=\s*(.*)", line)
            if m:
                var_name = m.group(1)
                value = m.group(2).strip()
                if value.endswith("\\"):
                    current_var = var_name
                    current_val = value[:-1].strip()
                else:
                    variables[var_name] = value

    # Handle unterminated continuation
    if current_var is not None:
        variables[current_var] = current_val.strip()

    return variables


def _resolve_makefile_var(
    value: str,
    variables: dict[str, str],
    depth: int = 0,
) -> str:
    """Resolve $(VAR) references in a Makefile value string.

    Handles simple single-level and recursive resolution up to depth 5.
    Replaces $(OBJEXT) with 'o' as a special case.
    """
    if depth > 5 or "$(" not in value:
        return value

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        if var_name == "OBJEXT":
            return "o"
        if var_name == "EXEEXT":
            return ""
        resolved = variables.get(var_name, "")
        return _resolve_makefile_var(resolved, variables, depth + 1)

    return re.sub(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)", _replace, value)


def _parse_autotools_link_units(
    build_dir: Path,
    executables: list[str],
    verbose: bool = False,
) -> tuple[list[dict], list[str]]:
    """Parse autotools Makefiles to discover executable link units.

    For each ELF executable found on disk, finds the corresponding Makefile
    in the same subdirectory and extracts am_<target>_OBJECTS and <target>_LDADD.

    Args:
        build_dir: Path to the build directory
        executables: List of executable paths (relative to build_dir)
        verbose: Print progress

    Returns:
        Tuple of (resolved_units, unresolved_exes):
        - resolved_units: list of link-unit dicts with objects and link_deps
        - unresolved_exes: list of executable paths we couldn't resolve
    """
    resolved = []
    unresolved = []

    # Group executables by directory (one Makefile per subdir)
    by_dir: dict[str, list[str]] = defaultdict(list)
    for exe in executables:
        exe_dir = str(Path(exe).parent)
        by_dir[exe_dir].append(exe)

    # Process each directory
    for exe_dir, exes in by_dir.items():
        makefile_path = build_dir / exe_dir / "Makefile"
        variables = _parse_makefile_variables(makefile_path)

        if not variables:
            if verbose:
                print(f"[heuristic] No Makefile in {exe_dir}")
            unresolved.extend(exes)
            continue

        for exe_rel in exes:
            exe_name = Path(exe_rel).stem  # e.g., "addr2line", "ld-new"

            # Autotools uses underscores in variable names, so convert
            # hyphens: "ld-new" -> try "ld_new" as the Makefile variable prefix
            target_names = [exe_name]
            if "-" in exe_name:
                target_names.append(exe_name.replace("-", "_"))

            found = False
            for target in target_names:
                # Look for am_<target>_OBJECTS
                obj_var = f"am_{target}_OBJECTS"
                obj_value = variables.get(obj_var, "")

                if not obj_value:
                    # Also try <target>_OBJECTS directly
                    obj_var = f"{target}_OBJECTS"
                    obj_value = variables.get(obj_var, "")

                if not obj_value:
                    continue

                # Resolve variable references
                obj_value = _resolve_makefile_var(obj_value, variables)

                # Parse object files
                target_objects = []
                for token in obj_value.split():
                    token = token.strip()
                    if not token or token.startswith("#"):
                        break
                    # Normalize: some have $(OBJEXT) already resolved to .o
                    if not token.endswith(".o"):
                        continue
                    # Make relative to build_dir
                    if not Path(token).is_absolute():
                        token = str(Path(exe_dir) / token)
                    target_objects.append(token)

                if not target_objects:
                    continue

                # Look for <target>_LDADD or <target>_DEPENDENCIES for link deps
                ldadd = _resolve_makefile_var(
                    variables.get(f"{target}_LDADD", ""), variables
                )
                deps_str = _resolve_makefile_var(
                    variables.get(f"{target}_DEPENDENCIES", ""), variables
                )

                link_deps = []
                # Parse library deps from LDADD/DEPENDENCIES
                for dep_source in (ldadd, deps_str):
                    for token in dep_source.split():
                        token = token.strip()
                        # Keep .a and .la library references
                        if token.endswith((".a", ".la")):
                            dep_name = Path(token).stem
                            if dep_name.startswith("lib"):
                                dep_name = dep_name[3:]  # libbfd -> bfd
                            if dep_name not in link_deps:
                                link_deps.append(dep_name)

                resolved.append({
                    "name": exe_name,
                    "type": "executable",
                    "output": exe_rel,
                    "objects": target_objects,
                    "link_deps": link_deps,
                })

                if verbose:
                    print(
                        f"[heuristic]   {exe_name}: "
                        f"{len(target_objects)} objects, "
                        f"deps={link_deps}"
                    )
                found = True
                break

            if not found:
                unresolved.append(exe_rel)
                if verbose:
                    print(f"[heuristic]   {exe_name}: unresolved")

    return resolved, unresolved


def _build_basename_index(
    compile_commands_path: Path | None,
) -> dict[str, str]:
    """Build an index from object basename to full output path.

    Maps e.g. "aio.lo" -> "/data/.../obj/src/aio/aio.c.lo"
    so that flat ar-t member names can be resolved to real paths.

    When multiple entries share a basename, the first one wins (ambiguous
    members will fall through to bc-mapping later).
    """
    index: dict[str, str] = {}
    if not compile_commands_path or not compile_commands_path.exists():
        return index
    try:
        with open(compile_commands_path) as f:
            entries = json.load(f)
        for entry in entries:
            output = entry.get("output")
            if not output:
                continue
            basename = Path(output).name
            if basename not in index:
                index[basename] = output
    except (json.JSONDecodeError, OSError):
        pass
    return index


def discover_heuristic(
    build_dir: Path,
    verbose: bool = False,
    compile_commands_path: Path | None = None,
) -> tuple[dict, list[str]]:
    """Heuristic link-unit discovery for non-Ninja builds.

    Combines prescan results with Makefile parsing:
    1. Libraries: ar t on every .a file
    2. Executables: parse autotools Makefiles for am_<target>_OBJECTS

    Args:
        build_dir: Path to the build directory
        verbose: Print progress
        compile_commands_path: Optional path to compile_commands.json for
            resolving ar-t member names to real output paths

    Returns:
        Tuple of (result_dict, unresolved_objects):
        - result_dict: partial link_units.json structure
        - unresolved_objects: .o paths that need agent help
    """
    build_dir = Path(build_dir).resolve()

    if verbose:
        print("[heuristic] Running prescan...")
    prescan = prescan_build_dir(build_dir, verbose=verbose)

    # Build basename index for resolving flat ar-t names to real paths
    basename_index = _build_basename_index(compile_commands_path)
    if verbose and basename_index:
        n = len(basename_index)
        print(
            f"[heuristic] Loaded {n} compile_commands outputs"
            f" for member resolution"
        )

    link_units = []

    # Step 1: Libraries from .a files
    if verbose:
        print(f"[heuristic] Processing {len(prescan['archives'])} archives...")
    seen_archives: dict[str, str] = {}  # filename -> first archive_rel (dedup)
    for archive_rel in prescan["archives"]:
        archive_path = build_dir / archive_rel
        members = run_ar_t(archive_path)
        if not members:
            continue

        # Derive library name from filename
        archive_filename = Path(archive_rel).name  # e.g., "libbfd.a"
        name = Path(archive_rel).stem  # e.g., "libbfd"
        if name.startswith("lib"):
            name = name[3:]  # strip "lib" prefix

        # Deduplicate: skip if we already processed an archive with the same filename
        if archive_filename in seen_archives:
            if verbose:
                dup = seen_archives[archive_filename]
                print(
                    f"[heuristic]   {name}: "
                    f"skipped (duplicate of {dup})"
                )
            continue
        seen_archives[archive_filename] = archive_rel

        # Resolve member names to real output paths via compile_commands
        objects = []
        resolved_count = 0
        for m in members:
            real_path = basename_index.get(m)
            if real_path:
                objects.append(real_path)
                resolved_count += 1
            else:
                # Fallback: path relative to archive directory
                archive_dir = str(Path(archive_rel).parent)
                objects.append(str(Path(archive_dir) / m))

        link_units.append({
            "name": name,
            "type": "static_library",
            "output": archive_rel,
            "objects": objects,
            "link_deps": [],
        })

        if verbose:
            if resolved_count:
                print(
                    f"[heuristic]   {name}: "
                    f"{len(members)} objects "
                    f"({resolved_count} resolved via "
                    f"compile_commands)"
                )
            else:
                print(f"[heuristic]   {name}: {len(members)} objects")

    # Step 2: Executables from ELF binaries + Makefile parsing
    if verbose:
        print(f"[heuristic] Processing {len(prescan['executables'])} executable candidates...")
    exe_units, unresolved = _parse_autotools_link_units(
        build_dir, prescan["executables"], verbose=verbose,
    )
    link_units.extend(exe_units)

    result = {
        "build_system": "autotools",
        "build_dir": str(build_dir),
        "link_units": link_units,
    }

    return result, unresolved


def discover_deterministic(
    build_dir: Path,
    compile_commands_path: Path | None = None,
    project_name: str | None = None,
    project_path: Path | None = None,
    verbose: bool = False,
) -> dict | None:
    """Fast-path deterministic link-unit discovery for Ninja builds.

    If build.ninja exists, parses it fully without any LLM calls.
    Returns link_units.json structure or None if not a Ninja build.

    Args:
        build_dir: Path to the build directory
        compile_commands_path: Optional path to compile_commands.json
        project_name: Project name (defaults to build_dir parent name)
        project_path: Project source path
        verbose: Print progress info

    Returns:
        Dict matching link_units.json format, or None if no build.ninja
    """
    build_ninja = build_dir / "build.ninja"
    if not build_ninja.exists():
        return None

    if project_name is None:
        project_name = build_dir.name

    if verbose:
        print(f"[link-units] Parsing {build_ninja}")

    # Parse ninja targets
    result = parse_ninja_targets(build_ninja)
    if "error" in result:
        if verbose:
            print(f"[link-units] Error: {result['error']}")
        return None

    targets = result["targets"]
    if not targets:
        if verbose:
            print("[link-units] No link targets found in build.ninja")
        return None

    if verbose:
        print(f"[link-units] Found {len(targets)} link targets")

    # Load compile_commands.json if available
    cc_entries = None
    if compile_commands_path and compile_commands_path.exists():
        try:
            with open(compile_commands_path) as f:
                cc_entries = json.load(f)
            if verbose:
                print(f"[link-units] Loaded {len(cc_entries)} compile_commands entries")
        except (json.JSONDecodeError, OSError) as e:
            if verbose:
                print(f"[link-units] Warning: Failed to load compile_commands: {e}")

    # Map objects to .bc files for each target
    link_units = []
    for target in targets:
        bc_result = map_objects_to_bc(
            target["objects"],
            build_dir,
            compile_commands=cc_entries,
        )

        bc_files = [
            bc for bc in bc_result["mappings"].values()
            if bc is not None
        ]
        asm_sources = bc_result.get("asm_sources", [])

        link_unit = {
            "name": target["name"],
            "type": target["type"],
            "output": target["output"],
            "objects": target["objects"],
            "bc_files": bc_files,
            "asm_sources": asm_sources,
            "link_deps": target["link_deps"],
        }
        link_units.append(link_unit)

        if verbose:
            s = bc_result["stats"]
            asm_str = f", {len(asm_sources)} asm sources" if asm_sources else ""
            print(
                f"[link-units]   {target['name']}: "
                f"{len(target['objects'])} objects, "
                f"{len(bc_files)} bc files "
                f"(tier1={s['tier1']}, tier2={s['tier2']}, missing={s['not_found']})"
                f"{asm_str}"
            )

    return {
        "project": project_name,
        "build_system": "cmake",
        "build_dir": str(build_dir),
        "link_units": link_units,
    }
