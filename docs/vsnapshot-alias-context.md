# V-Snapshot Alias Context for Contract Propagation

This document describes integrating whole-program aliasing information from
CFL-reachability V-snapshots into the memsafe (Pass 4) and verification (Pass 5)
LLM prompts, enabling LLMs to resolve field-level aliasing that formal analysis
cannot track.

## Problem

Bottom-up contract propagation fails when struct fields are aliased through
initializer functions. Consider zlib's `deflate_state`:

```c
// _tr_init sets up the alias:
void _tr_init(deflate_state *s) {
    s->bl_desc.dyn_tree = s->bl_tree;   // field alias
    s->bl_desc.stat_desc = &static_bl_desc;
}

// build_bl_tree operates through the alias:
void build_bl_tree(deflate_state *s) {
    // accesses s->bl_desc.dyn_tree (which IS s->bl_tree)
    // generates contract: bl_desc.dyn_tree: buffer_size(2*BL_CODES)
}

// _tr_flush_block calls build_bl_tree:
void _tr_flush_block(deflate_state *s, ...) {
    build_bl_tree(s);
    // LLM sees callee contract on "bl_desc.dyn_tree"
    // but locally uses "s->bl_tree" — doesn't connect them
    // propagates only not_null instead of buffer_size(2*BL_CODES)
}
```

The memsafe pass generates the correct contract for `build_bl_tree`
(`bl_desc.dyn_tree: buffer_size(2*BL_CODES)`), but `_tr_flush_block` cannot
propagate it because it doesn't know `bl_desc.dyn_tree == bl_tree`. The LLM
never sees `_tr_init` and has no reason to make the connection.

More broadly, aliasing arises through multiple channels beyond function
arguments:

- **Globals**: a global struct pointer may alias a function's parameter
  (e.g., a singleton instance passed to helpers)
- **Return values**: a factory function returns a pointer that aliases an
  argument's sub-object (e.g., `get_stream(ctx)` returns `ctx->strm`)
- **Cross-function dataflow**: function A stores into a global that function B
  reads — their contracts interact through the shared pointer

## V-Snapshot as Alias Substrate

The CFL-reachability analysis computes whole-program value aliasing (the
V-relation) and serializes it as a `.vsnap` file per link unit. The V-snapshot
is **field-insensitive** — it tracks pointer-level may-aliasing but cannot
directly express struct field relationships like `s->bl_tree == s->bl_desc.dyn_tree`.

The V-snapshot provides aliasing for all pointer-bearing program entities:

### Named Entry Kinds (from vsnapshot.py)

| Kind | Meaning | Name format | Aliasing use |
|------|---------|-------------|--------------|
| 1 | Function | `funcname` | Function pointer aliasing (indirect calls) |
| 2 | Global | `globalname` | Global pointers aliasing with args/returns |
| 3 | Argument | `funcname::arg#N` | Parameter aliasing across functions |
| 6 | Return | `ret:funcname` | Return value aliasing with args/globals |
| 7 | Vararg | `funcname::vararg` | Variadic argument aliasing |

All of these participate in the V-relation. An alias query like
`may_alias_name("ret:create_ctx", "process::arg#0")` tells us whether the
pointer returned by `create_ctx` may alias the first argument of `process` —
meaning contracts on the return value of one apply to the parameter of the
other.

### Query API

```python
from tools.vsnapshot import VSnapshot

snap = VSnapshot.load("func-scans/zlib/zlibstatic/zlibstatic.vsnap")

# Argument aliasing: do _tr_init and build_bl_tree share a pointer arg?
snap.may_alias_name("_tr_init::arg#0", "build_bl_tree::arg#0")  # True

# Return-to-arg aliasing: does deflateInit2's return alias deflate's arg?
snap.may_alias_name("ret:deflateInit2_", "deflate::arg#0")

# Global-to-arg aliasing: does a global alias a function's parameter?
snap.may_alias_name("g_context", "process_request::arg#0")

# Find all named entries aliasing a given node
nodes = snap.resolve_name("build_bl_tree::arg#0")
for node in nodes:
    for alias_rep in snap.aliases_of_node(node):
        # look up named entries with this rep
        ...
```

### What Aliasing Tells Us

The V-snapshot answers: "do these two pointer-level entities refer to the same
underlying object?" This is useful in several scenarios:

1. **Arg↔Arg**: Two functions receive the same struct pointer → an initializer
   among them may establish field aliases relevant to the other's contracts.
2. **Global↔Arg**: A global variable aliases a function parameter → the global's
   state (set by other functions) affects the function's contracts.
3. **Ret↔Arg**: A function's return value aliases another's argument → the
   returned pointer carries contracts that apply at the call site.
4. **Ret↔Global**: A factory's return value aliases a global → contracts on
   the returned object apply to the global and vice versa.

The V-snapshot identifies **which** entities alias; the LLM reads the relevant
source code to understand **how** (field assignments, initialization patterns).

## Design

### AliasContextBuilder

A new module `src/llm_summary/alias_context.py` that:

1. Loads a V-snapshot file (reusing `kanalyzer/tools/vsnapshot.py` or a
   vendored copy).
2. Builds a multi-kind alias index from all named entries (args, globals,
   returns) — grouping entries into alias equivalence classes by their
   V-relation representative.
3. Given a target function, queries the index across all entry kinds to find
   **related functions** — those whose args, return values, or associated
   globals alias with the target or its callees.
4. Among related functions, identifies **context-providing candidates**:
   initializers, factories, and global-setup functions whose source reveals
   field-level aliasing relationships.
5. Returns a formatted context block with the candidate source, annotated
   with the specific alias relationships discovered.

### Alias Index Structure

The index maps each named entry to its alias equivalence class and tracks
which functions participate:

```python
# Per alias equivalence class (V-relation representative):
@dataclass
class AliasGroup:
    rep: int                           # V-relation representative ID
    entries: list[NamedEntry]          # all named entries in this class
    functions: set[str]                # function names involved
    has_globals: bool                  # any global entries?
    has_returns: bool                  # any return entries?

# Index lookups:
func_to_groups: dict[str, list[AliasGroup]]   # function → its alias groups
global_to_groups: dict[str, list[AliasGroup]]  # global → its alias groups
```

This allows querying: "given function F, find all alias groups it participates
in (through any of its args, return value, or associated globals), and from
those groups, find other functions whose source may reveal aliasing structure."

### Candidate Selection

Not every aliased function provides useful context. We select candidates that
are likely to reveal field-level aliasing:

**Initializer-like functions** (primary candidates):
- Name pattern: `*init*`, `*create*`, `*setup*`, `*new*`, `*open*`, `*alloc*`
- Content signal: contains `->` field store patterns (struct field assignments)
- Relationship: shares at least one alias group with the target or its callees

**Factory/accessor functions** (secondary candidates):
- Return value aliases an argument's sub-object (`ret:func` ↔ `func::arg#N`)
- Source shows the return expression accesses a field of the argument
  (e.g., `return ctx->stream;`)

**Global-setup functions** (tertiary candidates):
- Write to a global that aliases a target function's parameter
- Source shows global assignment patterns

When multiple candidates exist, include all (bounded by a configurable limit,
default 3) sorted by relevance (number of shared alias groups, priority by
candidate type).

### Alias Annotation in Context

The context section annotates each candidate with the specific alias
relationships found, so the LLM knows exactly which pointers correspond:

```
### `_tr_init` — aliases with target function:
  - _tr_init::arg#0  ↔  _tr_flush_block::arg#0  (deflate_state *s)
  - _tr_init::arg#0  ↔  build_bl_tree::arg#0     (callee, same pointer)
```

For return-value aliasing:
```
### `create_context` — return value aliases:
  - ret:create_context  ↔  process::arg#0  (returned pointer is passed to process)
```

For global aliasing:
```
### `init_global_state` — global aliases:
  - g_state (global)  ↔  handle_request::arg#0  (global passed as parameter)
```

### Fallback

When no V-snapshot is available (e.g., legacy single-DB projects without CFL
analysis), the alias context section is omitted. The passes work exactly as
before — no behavioral change without `--vsnap`.

## Integration Points

### Pass 4: Memsafe Contract Generation

Add a new prompt section `{alias_context}` after the callee note:

```
## Alias Context (from whole-program analysis)

The following function(s) operate on pointers that alias with this function's
parameters, return value, or relevant globals. They may establish field-level
relationships relevant for understanding buffer sizes and pointer validity.

### `_tr_init` — aliases with target function:
  - _tr_init::arg#0  ↔  _tr_flush_block::arg#0  (same struct pointer)
  - _tr_init::arg#0  ↔  build_bl_tree::arg#0     (callee, same pointer)

```c
void _tr_init(deflate_state *s) {
    s->l_desc.dyn_tree = s->dyn_ltree;
    s->d_desc.dyn_tree = s->dyn_dtree;
    s->bl_desc.dyn_tree = s->bl_tree;
    ...
}
` ``

When a callee's contract references a field path (e.g., `bl_desc.dyn_tree`),
check whether an aliased function establishes it as an alias for another field
(e.g., `bl_tree`). If so, propagate the contract to the aliased field.

Also consider return-value and global aliasing: if a factory's return value
aliases a parameter here, contracts on the return apply to that parameter.
```

**Changes to `memsafe_summarizer.py`:**

- `summarize_function()` accepts optional `alias_context: str | None`
- If provided, inject into prompt via `{alias_context}` format slot
- `_build_alias_context()` delegates to `AliasContextBuilder`

**Changes to `MemsafePass` (driver.py):**

- Accept optional `AliasContextBuilder` in constructor
- In `summarize()`, call builder to generate context for the current function
- Thread through to `summarizer.summarize_function()`

### Pass 5: Verification

Add `{alias_context}` to the verification prompt, same content. This helps the
verifier understand when a callee contract target under an aliased name is
actually satisfied through the original field, or when a global's state
established elsewhere satisfies a local contract:

```
## Alias Context

[same format as Pass 4]

Use this to determine whether:
- A callee's pre-condition on an aliased field (e.g., `bl_desc.dyn_tree:
  buffer_size(38)`) is satisfied by a check on the original field (e.g.,
  `bl_tree` allocated with size >= 38).
- A return value's contract is established by the factory that created it.
- A global's invariant is maintained by setup code elsewhere.
```

**Changes to `verification_summarizer.py`:**

- `summarize_function()` accepts optional `alias_context: str | None`
- If provided, inject into prompt via `{alias_context}` format slot

**Changes to `VerificationPass` (driver.py):**

- Same pattern as MemsafePass: accept optional `AliasContextBuilder`

### CLI (`cli.py`)

Add `--vsnap` option to the `summarize` command:

```
--vsnap PATH    Path to V-snapshot (.vsnap) file for alias context
```

When provided, construct `AliasContextBuilder` and pass to relevant passes.

### Batch Scripts (`batch_summarize.py`)

For link-unit projects, `link_units.json` already has the `"vsnapshot"` field
per target (written by `batch_call_graph_gen.py`). Thread it through:

```python
vsnap_path = lu.get("vsnapshot")
if vsnap_path and Path(vsnap_path).exists():
    cmd += ["--vsnap", vsnap_path]
```

## Implementation Sketch

```python
# src/llm_summary/alias_context.py

from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class AliasGroup:
    """An equivalence class of aliased named entries."""
    rep: int
    entries: list  # list[NamedEntry]
    functions: set = field(default_factory=set)
    globals_: set = field(default_factory=set)
    returns: set = field(default_factory=set)


class AliasContextBuilder:
    """Builds alias context sections from V-snapshot data."""

    def __init__(self, vsnap_path: str, db: SummaryDB, max_candidates: int = 3):
        self.snap = VSnapshot.load(vsnap_path)
        self.db = db
        self.max_candidates = max_candidates
        self._groups: list[AliasGroup] | None = None
        self._func_to_groups: dict[str, list[AliasGroup]] | None = None

    def _build_index(self) -> None:
        """Build alias groups from ALL named entry kinds."""
        # Group named entries by V-relation representative
        rep_to_entries: dict[int, list] = defaultdict(list)
        for entry in self.snap.named_entries:
            rep = self.snap.rep(entry.node)
            rep_to_entries[rep].append(entry)

        self._groups = []
        self._func_to_groups = defaultdict(list)

        for rep, entries in rep_to_entries.items():
            group = AliasGroup(rep=rep, entries=entries)
            for e in entries:
                if e.kind == 3:  # arg: "funcname::arg#N"
                    func_name = e.name.split("::")[0]
                    group.functions.add(func_name)
                elif e.kind == 6:  # return: "ret:funcname"
                    func_name = e.name.split(":", 1)[1]
                    group.functions.add(func_name)
                    group.returns.add(func_name)
                elif e.kind == 2:  # global
                    group.globals_.add(e.name)
                elif e.kind == 7:  # vararg: "funcname::vararg"
                    func_name = e.name.split("::")[0]
                    group.functions.add(func_name)

            # Only keep groups with >1 participant (args/globals/returns)
            if len(group.functions) + len(group.globals_) > 1:
                self._groups.append(group)
                for func_name in group.functions:
                    self._func_to_groups[func_name].append(group)

    def build_context(self, func: Function, callee_names: list[str]) -> str | None:
        """Build alias context for func + its callees.

        Queries all alias groups (arg, global, return) to find related
        functions whose source reveals field-level aliasing.

        Returns formatted prompt section or None if no relevant aliases found.
        """
        if self._func_to_groups is None:
            self._build_index()

        # Collect alias groups for target + callees
        query_names = {func.name} | set(callee_names)
        relevant_groups: list[AliasGroup] = []
        for name in query_names:
            relevant_groups.extend(self._func_to_groups.get(name, []))

        # Find candidate functions from those groups
        candidates: set[str] = set()
        for group in relevant_groups:
            candidates |= group.functions
        candidates -= query_names  # exclude self and direct callees

        # Filter and rank candidates
        scored = []
        for cand_name in candidates:
            cand_func = self.db.get_function_by_name(cand_name)
            if cand_func and self._is_context_provider(cand_func):
                score = self._relevance_score(cand_name, relevant_groups)
                scored.append((score, cand_func))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        selected = [f for _, f in scored[:self.max_candidates]]

        return self._format_context(func, selected, relevant_groups)

    def _is_context_provider(self, func: Function) -> bool:
        """Check if function is likely to reveal aliasing structure."""
        name = func.name.lower()
        # Initializer/factory patterns
        if any(p in name for p in ["init", "create", "setup", "new", "open", "alloc"]):
            return True
        # Contains field store patterns
        if func.source and "->" in func.source and "=" in func.source:
            return True
        return False

    def _relevance_score(self, func_name: str, groups: list[AliasGroup]) -> int:
        """Score candidate by number of shared alias groups."""
        return sum(1 for g in groups if func_name in g.functions)

    def _format_context(
        self, target: Function, candidates: list[Function],
        groups: list[AliasGroup],
    ) -> str:
        """Format alias context section with alias annotations."""
        ...
```

## Data Flow

```
V-snapshot (.vsnap)
       │
       ▼
AliasContextBuilder
  ├─ _build_index()           ← alias groups from ALL named entry kinds
  │    ├─ kind=3 (arg)          (arg#N ↔ arg#N across functions)
  │    ├─ kind=2 (global)       (global ↔ arg/return aliasing)
  │    ├─ kind=6 (return)       (ret:func ↔ arg/global aliasing)
  │    └─ kind=7 (vararg)       (vararg ↔ arg/global aliasing)
  ├─ _is_context_provider()   ← initializer/factory/field-store heuristic
  └─ build_context()          ← formatted prompt section with annotations
       │
       ▼
  ┌────┴────┐
  │         │
  ▼         ▼
Pass 4    Pass 5
memsafe   verify
prompt    prompt
```

## Expected Impact

### Field Aliasing (arg↔arg)

Using the zlib `bl_tree` case:

**Before:** `_tr_flush_block` propagates `s->bl_tree: not_null` instead of
`buffer_size(2*BL_CODES)` because the LLM doesn't know
`bl_desc.dyn_tree == bl_tree`.

**After:** The prompt includes `_tr_init`'s source showing
`s->bl_desc.dyn_tree = s->bl_tree`. The LLM maps `build_bl_tree`'s contract
on `bl_desc.dyn_tree` to `bl_tree` and propagates `buffer_size(2*BL_CODES)`.

### Factory Return Values (ret↔arg)

When `create_stream()` returns a pointer that aliases `process_stream::arg#0`,
the LLM can see that contracts established by the factory (e.g., returned
buffer has capacity N) carry over to the consumer's parameter.

### Global State (global↔arg)

When a global `g_ctx` aliases `handler::arg#0`, the LLM can see how the global
is initialized elsewhere and apply those invariants when verifying `handler`.

This pattern applies broadly to C codebases using descriptor structs, factory
patterns, or global singletons (zlib, OpenSSL, SQLite, libpng, etc.).

## Scope and Limitations

- **Field-insensitive foundation**: V-snapshot confirms pointer-level aliasing
  but not field-level. The LLM must read the source to make field connections.
  This is a best-effort heuristic, not a formal guarantee.
- **Candidate heuristic**: May miss non-conventionally-named context providers
  or include false positives. The bounded limit (default 3) caps prompt bloat.
- **Prompt size**: Each included function adds its full source to the prompt.
  For very large functions, consider truncating to relevant field-assignment
  or return-expression lines only.
- **No V-snapshot**: Projects without CFL analysis get no alias context — the
  feature degrades gracefully to current behavior.
