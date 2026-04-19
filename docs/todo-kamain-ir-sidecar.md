# TODO: KAMain IR Fact Sidecar (Phase 2 of `design-llm-first.md`)

Tracking item for work that lives in the **KAMain repo**, not this one.
Phase 3 of the redesign (forward contract pass) consumes its output.

## Hard constraint

**KAMain does not touch llm-summary's databases.** No SQLite writes, no
schema migrations, no coupling to `functions.db` or
`function_features`. Output is **JSON only**. The llm-summary side owns
all persistence; it loads the JSON and decides what to store.

Rationale: KAMain is a separate analysis tool. Coupling it to a
sibling project's schema makes both repos harder to change. JSON
sidecar is a stable, language-agnostic interface.

## Scope (KAMain repo)

Extend the existing CFL/V-snapshot pass with a per-function IR walk that
emits a JSON sidecar. Single linear pass per function; reuses the
bitcode and CG already produced. Not a new tool — additions to the
existing pass.

## Output

- **Location**: one JSON file per `.bc`, sidecar named `<bc>.facts.json`
  (or one combined file per link unit — pick whichever matches existing
  KAMain output conventions).
- **No DB writes from KAMain.** The llm-summary repo gets a separate
  *sidecar loader* component that reads these files into whatever
  storage Phase 3 needs.

## Schema (per function)

Mirrors what `docs/design-llm-first.md` Phase 2 expects. Stable ids on
every fact so Phase 3 can reference symbolically rather than restate.

```jsonc
{
  "function": "ssl3_connect",
  "ir_hash": "<sha of function bitcode bytes>",  // for incremental invalidation
  "cg_hash": "<sha of sorted callee-id list>",   // for transitive feature invalidation

  // Enumerated effects: alloc, free, write, read, call, return, assume.
  // Each gets a stable id Phase 3 LLM annotates with semantics.
  "effects": [
    { "id": "e1", "kind": "alloc", "via": "malloc", "size_expr": "n",
      "target_ssa": "%out", "guard_bb": "bb3", "loc": "f.c:12" },
    { "id": "e2", "kind": "free",  "via": "free",   "target_ssa": "%p",
      "guard_bb": "bb7", "loc": "f.c:88" },
    { "id": "e3", "kind": "write", "target": "*%out", "value": "%src[0..n)",
      "guard_bb": "bb4", "loc": "f.c:15" },
    { "id": "e4", "kind": "call",  "callsite_id": "cs1", "callee": "memcpy",
      "args_ssa": ["%out", "%src", "%n"], "loc": "f.c:15" },
    { "id": "e5", "kind": "return", "value_ssa": "%out", "guard_bb": "bb6" }
  ],

  // Branches with conditions in source-readable form when feasible.
  "branches": [
    { "id": "b1", "loc": "f.c:10", "cond": "!s",
      "true_bb": "bb1", "false_bb": "bb2" }
  ],

  // Simple decidable numeric ranges (LLVM ConstantRange / SCEV). Where
  // the IR can prove a bound cheaply, record it. Don't reach for
  // anything fancy — Phase 3 LLM handles the rest.
  "ranges": [
    { "id": "r1", "ssa": "%n", "lo": 1, "hi": "INT_MAX" },
    { "id": "r2", "ssa": "%i", "lo": 0, "hi": "%n - 1", "scope_bb": "bb_loop" }
  ],

  // Alias annotations from V-snapshot. Phase 3 uses these to discharge
  // disjoint(p, q) / aliases(p, q) preconditions without re-asking LLM.
  "aliases": [
    { "id": "a1", "ssa": "%out", "points_to": ["heap_alloc_e1"], "loc": "f.c:12" },
    { "id": "a2", "ssa": "%src", "points_to": ["param_src"],     "loc": "f.c:0" }
  ],

  // Callsite enumeration (separate from effects[] for fast lookup).
  "callsites": [
    { "id": "cs1", "callee": "memcpy", "loc": "f.c:15",
      "args_ssa": ["%out", "%src", "%n"], "indirect": false }
  ],

  // Pass-gating feature bitfield from `plan-pass-gating.md`. Was
  // originally a DB column; now lives in the sidecar.
  "features": {
    "local_bits": 0x4321,            // bitfield per plan-pass-gating.md
    "static_callee_bits": 0x4FFF,    // OR over callees, fixpoint
    "ptr_params": 2, "out_params": 1, "return_is_ptr": true,
    "is_trivial_wrapper": false
  }
}
```

### ID rules

- Ids are unique **per function** and stable across runs given the same
  IR (sort by IR position when assigning, or use a deterministic hash).
- Phase 3 LLM references ids in the `ContractRecord.trace[].effect_id`,
  `effect_id`, `callsite_id`, `discharge_id` fields.
- Cross-function references use `function_name + local_id`.

## Integer overflow assistance

The current code-contract pipeline reaches 0 FN on the sv-comp
ControlFlow.set, but a small residual of FPs traces to corner cases
the LLM cannot resolve from C source alone (usual arithmetic
conversions, integer promotion, implicit casts hidden by macros,
optimizer-proved dead branches). A parallel residual of FNs would
appear on bigger codebases where the LLM misses an overflow site
buried in macros or inline expansion. The IR has the right view of
both sides — both come "for free" from a single per-function walk.

### What KAMain emits to help

Add an `int_ops` array to the per-function sidecar (sibling of
`effects` / `branches` / `ranges`). Each entry records one IR-level
integer operation, with the IR-provable facts the LLM otherwise has
to derive (or invent) from source:

```jsonc
"int_ops": [
  // Signed arithmetic — overflow candidate. The frontend kept `nsw`,
  // so the operation is signed at the IR-promoted type.
  { "id": "io1", "op": "mul", "type": "i32", "nsw": true, "nuw": false,
    "lhs_ssa": "%a", "rhs_ssa": "%b",
    "lhs_range": "r3", "rhs_range": "r4",        // refs into ranges[]
    "promoted_from": ["i16", "i16"],             // operand types pre-promotion
    "loc": "f.c:42" },

  // Unsigned arithmetic — wraps modulo 2^N, well-defined, NOT a
  // candidate. Phase 3 must not flag this; the explicit
  // `wraps_legally: true` discharges the case.
  { "id": "io2", "op": "add", "type": "i32", "nsw": false, "nuw": false,
    "wraps_legally": true,
    "lhs_ssa": "%u", "rhs_ssa": "%v",
    "loc": "f.c:50" },

  // Mixed signed+unsigned of equal rank — usual arithmetic conversions
  // make this unsigned. Source reads as `s * u` where `s` is signed,
  // but IR shows the operation type is unsigned. Marker explicit so
  // Phase 3 can drop the false flag.
  { "id": "io3", "op": "mul", "type": "i32",
    "wraps_legally": true,
    "uac_unsigned": true,                        // result type forced unsigned
    "loc": "f.c:55" },

  // Division — discharged if rhs proven nonzero by a dominating
  // branch or SCEV. Phase 3 cites `rhs_nonzero` instead of
  // re-deriving the dominator chain.
  { "id": "io4", "op": "sdiv", "type": "i32",
    "lhs_ssa": "%x", "rhs_ssa": "%d",
    "rhs_nonzero": true,
    "rhs_range": "r5",
    "loc": "f.c:60" },

  // Shift — discharged if amount range proven within [0, bitwidth).
  { "id": "io5", "op": "shl", "type": "i32",
    "lhs_ssa": "%v", "amt_ssa": "%s",
    "amt_range": "r6",                           // ranges[r6] = [0, 31]
    "amt_in_range": true,
    "loc": "f.c:70" },

  // Sign-changing cast — `trunc i64 to i32` with src range fits in
  // i32 → no UB. Source range comes from ranges[r7].
  { "id": "io6", "op": "trunc", "from": "i64", "to": "i32",
    "src_ssa": "%w", "src_range": "r7",
    "src_fits_dst": true,
    "loc": "f.c:80" }
],

// Functions whose return value is treated as full-range untrusted
// input by Phase 3 (no need to re-derive). Both sv-comp helpers and
// real-world I/O sources go here; the source string is what the
// llm-summary stdlib map keys on.
"nondet_sources": [
  { "id": "ns1", "callee": "__VERIFIER_nondet_int",  "type": "i32" },
  { "id": "ns2", "callee": "scanf",                  "out_param": 1 }
]
```

### How this kills FPs

1. **Usual arithmetic conversions.** `int * unsigned` reads as signed
   in source; IR shows the operation type is unsigned and the
   sidecar marks `wraps_legally: true` + `uac_unsigned: true`. Phase
   3 prompt cites the entry id and drops the flag.
2. **Integer promotion.** `char a, b; a + b` becomes `add i32` after
   `sext`. Sidecar's `promoted_from: ["i8", "i8"]` + `type: "i32"`
   tells Phase 3 to check overflow at i32 range, not i8.
3. **Provable-nonzero divisors / in-range shifts.** SCEV / dominator
   analysis discharges these in the sidecar (`rhs_nonzero: true`,
   `amt_in_range: true`). Phase 3 doesn't re-derive.
4. **Boundary-representable results.** `65536 * -32768 == INT_MIN`:
   IR's ConstantRange computes `[INT_MIN, INT_MIN]`, fits in i32,
   no `nsw` violation. Sidecar's `lhs_range` + `rhs_range` carry
   the ranges; Phase 3 prompt's worked example becomes redundant.
5. **Dead branches.** If LLVM proved a branch unreachable, the
   `branches[]` entry can carry `unreachable_side: "true"` so Phase
   3 doesn't reason about ops gated by it.
6. **Unsigned-to-signed within-range casts.** `src_fits_dst: true`
   on the `trunc` / `zext` entry discharges the cast.

### How this kills FNs

1. **Exhaustive op enumeration.** Every signed arithmetic, signed
   div/rem, shift, and sign-changing cast in the function appears in
   `int_ops[]`. The LLM can't miss one buried in a macro expansion,
   inline-function expansion, or operator-overload site, because
   IR sees the post-inlined form.
2. **Untrusted-input lineage.** `nondet_sources[]` marks functions
   whose return values must be treated as full-range. Phase 3 walks
   `int_ops` def-use back to a `nondet_sources` entry to know which
   ops carry untrusted operands — rather than relying on the LLM to
   recognize every nondet helper by name.
3. **Promotion-revealed overflow.** `short s = INT_MAX; s + 1`: in
   source the `+` looks fine at short range, but IR shows
   `add nsw i32` after promotion. The `nsw` flag in the sidecar is
   the signal that the frontend believes overflow here is UB.
4. **Hidden conversions in macros.** Bit-field widening,
   `size_t`/`ssize_t` round-trips, container_of-style casts — all
   collapse to explicit `sext` / `zext` / `trunc` in IR with their
   own `int_ops` entries.

### Cost note

`int_ops` is one linear walk over each function's IR using existing
LLVM APIs (instruction iterator, `ConstantRange::fromConstantRange`,
`isKnownNonZero`, `computeKnownBits`, `ScalarEvolution`). No new
analyses required; piggyback on whatever `ranges[]` and `branches[]`
already cost. Sidecar size grows linearly with arithmetic-op count;
on libpng/openssl-scale code that's still small relative to the
`effects[]` block.

## Pass-gating features migration

`docs/plan-pass-gating.md` previously specified a `function_features`
SQL table written directly by KAMain. **Override**: KAMain emits the
same bitfield in the JSON sidecar's `features` block. The llm-summary
side decides whether to materialize a SQL table from the JSON for
query convenience — but that's a llm-summary concern, not KAMain's.

If `plan-pass-gating.md` and this doc disagree, **this doc wins**
(KAMain emits JSON; llm-summary owns SQL).

## Acceptance / validation

- Run on libpng, openssl, sv-comp `sv-control-flow` benchmarks.
- Spot-check 10 functions per project: every visible memory op, call,
  branch should appear in the sidecar with matching `loc`.
- Feature counts must match `plan-pass-gating.md` Section "Validation
  Plan" expectations (target: ≥50% skip rate on at least 3 passes).
- Sidecars must be deterministic across runs (byte-equal for same IR).

## llm-summary side (this repo)

Separate work item, not part of this TODO:

- **Sidecar loader** — reads `<bc>.facts.json`, exposes Phase 2 facts
  to Phase 3 by stable id. May materialize a `function_features` table
  for SQL gating queries if convenient.
- **Phase 3 forward contract pass** — consumes `body` (from libclang)
  + sidecar facts + callees' `ContractRecord`s; emits unified
  `ContractRecord` per function.

These are separate TODOs once the sidecar shape is locked.

## Out of scope

- Anything that requires LLVM passes to write SQL.
- Indirect call resolution — already handled by KAMain + LLM resolver
  (existing).
- Build-learn agent integration — orthogonal.
