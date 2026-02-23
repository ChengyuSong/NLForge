---
name: llm-summary
description: Reference and runner for llm-summary CLI commands. Use when the user asks to check stats, run summaries, look up functions, import call graphs, or do anything with a func-scans/<project>/functions.db.
argument-hint: "[command] [project] [options]"
---

# llm-summary CLI Reference

**Always activate the venv first:**
```bash
source ~/project/llm-summary/venv/bin/activate
```

**DB paths follow the pattern:** `func-scans/<project>/functions.db`

---

## Common Commands

### stats — check DB state
```bash
llm-summary stats --db func-scans/<project>/functions.db
```
Shows counts for functions, call_edges, allocation/free/init/memsafe/verification summaries.
**Run this first** to understand what's in the DB before doing anything else.

### summarize — run LLM summary passes
```bash
llm-summary summarize --db func-scans/<project>/functions.db \
  --type allocation --type free --type init \
  --backend gemini -v

llm-summary summarize --db func-scans/<project>/functions.db \
  --type memsafe \
  --backend gemini -v
```
- `--type`: `allocation`, `free`, `init`, `memsafe`, `verify` (can repeat)
- `--backend`: `claude`, `gemini`, `openai`, `ollama`, `llamacpp`
- `--model TEXT`: override model name
- `--llm-host / --llm-port`: for local backends (ollama: 11434, llamacpp: 8080)
- `--disable-thinking`: disable thinking mode for llamacpp
- `-f / --force`: re-summarize even if cached
- `--allocator-file PATH`: JSON file with custom allocator names
- `--deallocator-file PATH`: JSON file with custom deallocator names
- `--init-stdlib`: pre-populate stdlib summaries before starting
- `--log-llm PATH`: log all LLM prompts/responses to file
- Requires `call_edges > 0` — run `import-callgraph` first if needed
- Run `allocation + free + init` before `memsafe` (two separate invocations)

### lookup — look up a specific function
```bash
llm-summary lookup <function_name> --db func-scans/<project>/functions.db
llm-summary lookup <function_name> --db func-scans/<project>/functions.db --signature <sig>
```

### show — display summaries
```bash
llm-summary show --db func-scans/<project>/functions.db
llm-summary show --db func-scans/<project>/functions.db --name <function_name>
llm-summary show --db func-scans/<project>/functions.db --file <path_fragment>
llm-summary show --db func-scans/<project>/functions.db --allocating-only
llm-summary show --db func-scans/<project>/functions.db --format json
```

### import-callgraph — import KAMain JSON call graph
```bash
llm-summary import-callgraph --json func-scans/<project>/callgraph.json \
  --db func-scans/<project>/functions.db --clear-edges -v
```
- `--json PATH`: path to KAMain callgraph JSON (required)
- `--clear-edges`: clear existing call_edges before import
- `-v`: verbose output showing match stats

### init-stdlib — add stdlib summaries
```bash
llm-summary init-stdlib --db func-scans/<project>/functions.db
```

### import-dep-summaries — copy summaries from dep DBs
```bash
llm-summary import-dep-summaries \
  --db func-scans/<project>/<target>/functions.db \
  --dep-db func-scans/<dep_project>/<dep_target>/functions.db
```
- `--dep-db`: repeat for multiple deps
- `-f / --force`: overwrite existing summaries
- Imported summaries tagged with `model_used=dep:<dep_db_stem>`
- Idempotent without `--force`

### export — export summaries to JSON
```bash
llm-summary export --db func-scans/<project>/functions.db -o output.json
```

### callgraph — export call graph
```bash
llm-summary callgraph --db func-scans/<project>/functions.db
llm-summary callgraph --db func-scans/<project>/functions.db --format csv -o graph.csv
llm-summary callgraph --db func-scans/<project>/functions.db --format json -o graph.json
```
- `--format`: `tuples` (default), `csv`, `json`
- `--no-header`: omit header row (for csv/tuples)

### scan — extract functions from compile_commands.json (no LLM)
```bash
# Full project (legacy / no link-unit scoping)
llm-summary scan \
  --compile-commands build-scripts/<project>/compile_commands.json \
  --db func-scans/<project>/functions.db -v

# Per link-unit (preferred when link_units.json exists)
llm-summary scan \
  --compile-commands build-scripts/<project>/compile_commands.json \
  --link-units func-scans/<project>/link_units.json \
  --target <target_name> \
  --db func-scans/<project>/<target_name>/functions.db
```
- `--compile-commands PATH`: path to compile_commands.json (required)
- `--link-units PATH`: path to link_units.json (from `discover-link-units`); restricts scan to named target
- `--target TEXT`: link-unit target name (required with `--link-units`)
- `--project-path PATH`: host source root — **required** when compile_commands.json has Docker container paths (`/workspace/src/`); use the raw build-artifacts copy in that case. The `build-scripts/<project>/compile_commands.json` produced by `build-learn` already has host paths and does not need this flag.

**Prefer `build-scripts/<project>/compile_commands.json`** (host paths, produced by `build-learn`).
Use `/data/csong/build-artifacts/<project>/compile_commands.json` only when the build-scripts copy is missing, and add `--project-path /data/csong/opensource/<project>`.

### discover-link-units — discover per-target bc files
```bash
llm-summary discover-link-units \
  --build-dir /data/csong/build-artifacts/<project> \
  --project-name <project> \
  --output func-scans/<project>/link_units.json
```
Produces `link_units.json` with per-target `bc_files` lists used by `scan --link-units`.

### build-learn — learn how to build a project (ReAct agent)
```bash
llm-summary build-learn \
  --project-path /data/csong/opensource/<project> \
  --build-dir /data/csong/build-artifacts/<project> \
  --backend claude \
  --model claude-sonnet-4-5@20250929 \
  -v
```
- Runs a ReAct loop in a Docker container to figure out how to build the project
- Generates a reusable `build-scripts/<project>/build.sh`
- `--enable-lto / --no-lto`: control LLVM LTO (default: on)
- `--prefer-static / --no-static`: prefer static linking (default: on)
- `--generate-ir / --no-ir`: emit LLVM IR artifacts (default: on)
- `--max-retries INT`: max build attempts (default: 3)
- `--container-image TEXT`: Docker image (default: `llm-summary-builder:latest`)
- `--no-ccache`: disable ccache
- `--ccache-dir PATH`: host ccache directory (default: `~/.cache/llm-summary-ccache`)
- `--log-llm PATH`: log all LLM prompts/responses to file

### clear — wipe all data
```bash
llm-summary clear --db func-scans/<project>/functions.db
```

---

## Batch Scripts (in scripts/)

| Script | Purpose |
|--------|---------|
| `batch_scan_targets.py` | Scan all projects; link-unit-aware when `link_units.json` present |
| `batch_call_graph_gen.py` | Run KAMain + import call graphs; writes `db_path`/`callgraph_json`/`cflcg`/`vsnapshot` back to `link_units.json` |
| `batch_summarize.py` | Run allocation+free+init+memsafe passes; link-unit-aware when `link_units.json` present |
| `batch_verify.py` | Run verify pass; link-unit-aware when `link_units.json` present |
| `batch_rebuild.py` | Rebuild projects with debug info |

All batch scripts support: `--filter <name>`, `--tier <1|2|3>`, `--skip-list`, `--verbose`

`batch_summarize.py` and `batch_verify.py` also support: `--success-list`

`batch_call_graph_gen.py` also supports: `--compositional` / `--no-compositional`

---

## Analysis Pipeline Order

### Legacy (one DB per project)
1. **build-learn** — compile with LTO + `-save-temps=obj`, generate `build.sh`
2. **batch_scan_targets** — extract functions into `func-scans/<project>/functions.db`
3. **batch_call_graph_gen** — run KAMain, import edges
4. **batch_summarize** — Pass 1: `allocation+free+init`, Pass 2: `memsafe`
5. **batch_verify** — cross-pass verification

### Link-unit-aware (one DB per target)
1. **build-learn** — produces `build-scripts/<project>/compile_commands.json` (host paths)
2. **discover-link-units** — produces `func-scans/<project>/link_units.json`
3. **batch_scan_targets** — auto-detects `link_units.json`; scans each target into `func-scans/<project>/<target>/functions.db`
4. **batch_call_graph_gen** `--compositional` — per-target two-phase CFL; writes `db_path`, `callgraph_json`, `cflcg`, `vsnapshot` back into `link_units.json`
5. **batch_summarize** `--init-stdlib` — per target in topo order:
   - `import-dep-summaries` from intra-project dep DBs
   - `init-stdlib`
   - Pass 1: `allocation+free+init`, Pass 2: `memsafe`
6. **batch_verify** — per target in topo order

**Key files per link unit** (recorded in `link_units.json` after step 4):
- `db_path` — functions DB
- `callgraph_json` — KAMain call graph
- `cflcg` — compressed CFL constraint graph
- `vsnapshot` — serialized V-relation (value aliasing)
