# Link-Unit-Aware Analysis

## Problem

Projects often build multiple link units — libraries and executables — from a
single source tree. The current pipeline combines all bitcode files into one
KAMain invocation, which causes:

1. **Symbol collisions**: multiple `main` functions (one per executable)
   create ambiguous stubs in the DB.
2. **Polluted call graphs**: executable-specific code (test drivers, CLI
   tools) appears in the library analysis, adding noise.
3. **Wasted work**: library functions are re-analyzed for every project that
   depends on them (e.g. zlib summaries recomputed when analyzing libpng).
4. **Path mismatches**: container build paths vs host paths for the same
   function cause duplicate orphaned stubs.

## Design

### Key Principles

- **One DB per link unit.** Each library or executable gets its own
  `functions.db`. Cross-target references use imported summary stubs.
- **Dependency-ordered analysis.** Libraries are analyzed before the
  executables and libraries that depend on them.
- **Cross-project reuse.** A project's library summaries (e.g. zlib) can be
  imported as dependency stubs when analyzing a downstream project (e.g.
  libpng), without re-running the LLM.
- **KAMain summary export/import.** KAMain will support exporting reusable
  analysis results (alias info, call edges, points-to summaries) so that
  dependents don't need the full bitcode of their deps — only the exported
  summary.

### Directory Layout

```
func-scans/
  zlib/
    link_units.json                 # discovered link structure
    zlibstatic/
      functions.db                  # library analysis results
      callgraph.json                # KAMain output
      ka_export.json                # KAMain reusable summary (future)
    example/
      functions.db                  # executable analysis
      callgraph.json
  libpng/
    link_units.json
    libpng16/
      functions.db                  # depends on zlib/zlibstatic
      callgraph.json
    pngtest/
      functions.db                  # depends on libpng16
      callgraph.json
```

### link_units.json Format

Produced by the link-unit discovery agent. Describes all link units, their
bitcode files, and intra-project dependency relationships.

Top-level key is `"link_units"` (array). `bc_files` are **absolute host
paths**. Same-project deps are listed under `"link_deps"` (by output
filename). The `"objects"` field lists the `.o` members used to derive the
`bc_files`.

The analysis output fields (`db_path`, `callgraph_json`, `cflcg`,
`vsnapshot`) are **written back by `batch_call_graph_gen.py`** after
processing. They use absolute paths and may be `null` if the step did not
produce that artifact.

```json
{
  "project": "zlib",
  "build_system": "cmake",
  "build_dir": "/data/csong/build-artifacts/zlib",
  "link_units": [
    {
      "name": "zlibstatic",
      "type": "static_library",
      "output": "libz.a",
      "objects": [
        "CMakeFiles/zlibstatic.dir/adler32.c.o",
        "CMakeFiles/zlibstatic.dir/compress.c.o",
        "CMakeFiles/zlibstatic.dir/deflate.c.o"
      ],
      "bc_files": [
        "/data/csong/build-artifacts/zlib/CMakeFiles/zlibstatic.dir/adler32.bc",
        "/data/csong/build-artifacts/zlib/CMakeFiles/zlibstatic.dir/compress.bc",
        "/data/csong/build-artifacts/zlib/CMakeFiles/zlibstatic.dir/deflate.bc"
      ],
      "link_deps": [],
      "db_path": "func-scans/zlib/zlibstatic/functions.db",
      "callgraph_json": "func-scans/zlib/zlibstatic/callgraph.json",
      "cflcg": "func-scans/zlib/zlibstatic/zlibstatic.cflcg",
      "vsnapshot": "func-scans/zlib/zlibstatic/zlibstatic.vsnap"
    },
    {
      "name": "zlib_static_example",
      "type": "executable",
      "output": "test/zlib_static_example",
      "objects": [
        "test/CMakeFiles/zlib_static_example.dir/example.c.o"
      ],
      "bc_files": [
        "/data/csong/build-artifacts/zlib/test/CMakeFiles/zlib_static_example.dir/example.bc"
      ],
      "link_deps": ["libz.a"],
      "db_path": "func-scans/zlib/zlib_static_example/functions.db",
      "callgraph_json": "func-scans/zlib/zlib_static_example/callgraph.json",
      "cflcg": "func-scans/zlib/zlib_static_example/zlib_static_example.cflcg",
      "vsnapshot": "func-scans/zlib/zlib_static_example/zlib_static_example.vsnap"
    }
  ]
}
```

Cross-project dependencies (e.g. libpng → zlib) are **not yet recorded** in
`link_units.json`. They are tracked separately in the top-level
`project_deps.json` registry (see Batch Processing).

## Pipeline

### Phase 0: Build (existing)

`build-learn` agent produces `build.sh`, compiles with LTO +
`-save-temps=obj`, generates `compile_commands.json`.

**Note:** `compile_commands.json` is generated inside the Docker build
container and retains container-internal paths (`/workspace/src/`,
`/workspace/build/`). Downstream tools that consume it on the host must
remap these to the real host paths (see Phase 2).

### Phase 1: Discover Link Units (new)

A ReAct agent explores the build artifacts directory to identify all link
units and their dependency relationships.

**Input:** build directory after Phase 0.

**Agent skills (deterministic tools):**

| Skill | Description |
|-------|-------------|
| `detect_cmake_targets` | Parse `build.ninja` or `CMakeFiles/<t>.dir/link.txt` for link rules. Classify targets as library/executable from CMake target type. |
| `detect_autotools_targets` | Parse `Makefile` for `_LTLIBRARIES`, `_PROGRAMS`, `_LDADD`. Inspect `.la` libtool archives. |
| `detect_meson_targets` | Parse `build.ninja` meson-generated link rules. |
| `inspect_archive` | `ar t lib.a` → list members, map `.o` → `.bc` via `-save-temps` naming. |
| `inspect_elf` | `readelf -d` for dynamic deps; `nm --defined-only` for symbols. |
| `match_bc_files` | Given a list of `.o` files, find corresponding `.bc` files from `-save-temps=obj`. |

**Output:** `link_units.json` written to `func-scans/<project>/`.

The agent uses build-system knowledge to call the right skills. For unknown
build systems, it falls back to artifact inspection (archives, ELF headers,
`nm` for `main`-defining translation units).

### Phase 2: Scan Functions (existing, scoped per target)

`llm-summary scan` runs per link unit, producing per-target `functions.db`.

```bash
llm-summary scan \
  --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json \
  --link-units func-scans/zlib/link_units.json \
  --target zlibstatic \
  --project-path /data/csong/opensource/zlib \
  --db func-scans/zlib/zlibstatic/functions.db
```

- `--link-units` / `--target`: restrict extraction to source files whose
  corresponding `.bc` files appear in the named target's `bc_files` list.
  Without these options, all source files in `compile_commands.json` are
  scanned (original behaviour).
- `--project-path`: host path to the project source root. Required when
  `compile_commands.json` uses Docker container paths (`/workspace/src/`).
  Maps `/workspace/src` → `project-path` and `/workspace/build` → `build_dir`
  from `link_units.json`.

### Phase 3: Call Graph (modified)

#### Library targets (no external deps)

Run KAMain on the library's own bc_files:

```bash
KAMain --bc-list zlibstatic_bc.txt \
       --callgraph-json func-scans/zlib/zlibstatic/callgraph.json \
       --export-summary func-scans/zlib/zlibstatic/ka_export.json  # future
```

#### Library targets (with external deps)

Import dep summaries instead of sending dep bitcode:

```bash
KAMain --bc-list libpng16_bc.txt \
       --import-summary func-scans/zlib/zlibstatic/ka_export.json \
       --callgraph-json func-scans/libpng/libpng16/callgraph.json \
       --export-summary func-scans/libpng/libpng16/ka_export.json
```

`--import-summary` provides KAMain with pre-computed alias/call-edge info
for the dependency, avoiding the need for its bitcode files. KAMain treats
imported functions as opaque stubs with known call behavior.

**Fallback (before KAMain supports --import-summary):** pass dep bc_files
alongside target bc_files, same as today but scoped per-target.

#### Executable targets

Same as library-with-deps, but the executable's `main` is the entry point
and not a conflict since each executable is analyzed in its own DB.

### Phase 4: Import Call Graph (existing, per target)

```bash
llm-summary import-callgraph \
  --json func-scans/zlib/zlibstatic/callgraph.json \
  --db func-scans/zlib/zlibstatic/functions.db \
  --clear-edges
```

### Phase 5: Init Stdlib + Import Dep Summaries (modified)

Before summarization, populate the target DB with:

1. **Stdlib summaries** (via `init-stdlib`, using the global cache).
2. **Dependency summaries** — import from dep DB(s) so the bottom-up driver
   treats dep functions as pre-summarized leaves.

```bash
# Stdlib (existing)
llm-summary init-stdlib \
  --db func-scans/libpng/libpng16/functions.db \
  --backend gemini

# Dep summaries (new)
llm-summary import-dep-summaries \
  --db func-scans/libpng/libpng16/functions.db \
  --dep-db func-scans/zlib/zlibstatic/functions.db
```

`import-dep-summaries` copies function stubs and their summaries
(allocation, free, init, memsafe, verification) from the dep DB into the
target DB, tagged with `model_used="dep:<project>/<target>"`.

The bottom-up driver's cache check will find these and skip them during
summarization.

### Phase 6: Summarize (existing, incremental)

```bash
llm-summary summarize \
  --db func-scans/libpng/libpng16/functions.db \
  --type allocation --type free --type init \
  --backend gemini
```

The driver processes functions in topological order. Functions with
existing summaries (from stdlib, deps, or prior runs) are cache hits —
only new target-specific functions get LLM calls.

## Incremental Analysis Flow

### Example: libpng depends on zlib

```
Step 1: Analyze zlib (no deps)
  discover-link-units  → zlib/link_units.json
  scan (--link-units zlib/link_units.json --target zlibstatic
        --project-path /data/.../zlib)
                       → zlib/zlibstatic/functions.db
  KAMain               → zlib/zlibstatic/callgraph.json
                          zlib/zlibstatic/ka_export.json
  import-callgraph     → call_edges in functions.db
  init-stdlib          → stdlib stubs
  summarize            → all zlib functions summarized

Step 2: Analyze libpng (depends on zlib)
  discover-link-units  → libpng/link_units.json
  scan (--link-units libpng/link_units.json --target libpng16
        --project-path /data/.../libpng)
                       → libpng/libpng16/functions.db
  KAMain (--import-summary zlib/ka_export.json)
                       → libpng/libpng16/callgraph.json
  import-callgraph     → call_edges
  init-stdlib          → stdlib stubs
  import-dep-summaries → zlib function summaries copied in
  summarize            → only libpng-specific functions need LLM calls
                          zlib functions → cache hit, skipped
```

### Reanalysis

When zlib is updated and re-summarized:
1. Re-run the zlib pipeline (Phases 2–6).
2. For downstream projects (libpng), re-run `import-dep-summaries` to
   refresh the imported stubs, then re-run `summarize --force` on functions
   whose dep summaries changed.

## KAMain Summary Export/Import (Future)

KAMain will support:

| Flag | Description |
|------|-------------|
| `--export-summary PATH` | After analysis, write reusable results (function signatures, alias info, call edges, points-to summaries) to a JSON file. |
| `--import-summary PATH` | Before analysis, load a previously exported summary. Functions from the import are treated as opaque stubs with known behavior — no bitcode required. |

The export format captures enough information for KAMain to:
- Resolve calls from the current target into the dependency.
- Propagate alias/points-to information across the boundary.
- Report accurate call edges without re-analyzing dep bitcode.

**Granularity:** one export file per link unit. When a target has multiple
deps, multiple `--import-summary` flags are passed.

## Batch Processing

`batch_call_graph_gen.py` and `batch_summarize.py` are updated to:

1. Read `link_units.json` for each project.
2. Process targets in dependency order (topological sort within and across
   projects).
3. Pass per-target bc_files to KAMain (not all project bc_files).
4. Run `import-dep-summaries` before summarization.

A new `project_deps.json` registry at the top level records cross-project
dependencies:

```json
{
  "libpng": {"deps": ["zlib"]},
  "libtiff": {"deps": ["zlib", "libjpeg-turbo"]},
  "libavif": {"deps": ["libjpeg-turbo", "libpng", "zlib"]}
}
```

## Migration

The transition from one-DB-per-project to one-DB-per-link-unit is
incremental:

1. Projects without `link_units.json` continue to work as today (all
   bc_files → one KAMain run → one functions.db).
2. When `link_units.json` is present, the pipeline switches to per-target
   mode.
3. Existing `func-scans/<project>/functions.db` can be kept as-is or
   migrated by re-running the pipeline with link-unit awareness.
