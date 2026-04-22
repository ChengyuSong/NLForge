"""Per-property prompts for the code-contract pipeline.

Lifted from `scripts/contract_pipeline.py` SYSTEM_PROMPT, MEMSAFE_PROMPT,
MEMLEAK_PROMPT, OVERFLOW_PROMPT, plus the verify-pass prompts.

De-sv-comp edits applied at lift time (Step 2 of the plan):
  - SYSTEM_PROMPT rule 8: replaced "stdlib, `__VERIFIER_*`, etc." with
    "stdlib, intrinsics, harness helpers" — the message is the same
    (trust the inlined contract block) but does not bias the model toward
    sv-comp idioms.
  - VERIFY_PROMPT rule 5: dropped the `__VERIFIER_*` / `assume_abort_if_not`
    examples; kept the generic noreturn cue.

NO de-verbose edits applied. Today's A/B (`/tmp/cache_ab_findings.md`)
showed verbose per-property prompts win on small models (Qwen drops
"no resource acquired" trivial ensures with the short tail). Per the
`feedback_smaller_faster_models.md` memory, small/fast models are the
primary target — keep the verbose form everywhere. Backend-aware short
tails are deferred (see plan Open Items).
"""


SYSTEM_PROMPT = """\
You produce Hoare-style pre/post summaries for C functions, ONE PROPERTY AT A TIME.

## Hard rules

1. **Specific predicates; code-form preferred, prose OK when natural.** \
Each item must pin a CONCRETE fact about a named variable, field, position, \
or range. Code expressions in the source language (C or C++) are preferred \
when they fit (`p != NULL`, `0 <= i && i < n`, `len <= sizeof(buf)`, \
`setEventCalled == 1`). Compact value-range forms are also fine for \
integer ranges (`n: [0, INT_MAX]`, `i: [1, len-1]`). Prose is acceptable \
when it compresses the fact more cleanly than a code expression would \
(`s null-terminated; '\\0' at index length-1`, `p[0..length-1] initialized`, \
`result allocated for n elements`). Avoid BOTH formal-logic notation \
(`\\valid(p)`, `\\forall x`) AND over-compressed prose that hides specifics \
(`p is a valid string` — what does "valid" mean? what's the length? is it \
null-terminated and where?). The downstream consumer is another LLM call: \
clarity-for-that-reader is the bar, not formal-tool checkability.

2. **Caller-accessible names only — STRICT.** Every `requires` / `ensures` \
clause is read by the CALLER. The caller cannot see anything you declared \
inside the function body. Reference ONLY:
   - this function's parameter names (read them off the signature, not \
     the body);
   - the literal `result` for the return value;
   - out-parameter cells (`*out_p`, `out_p->field` — only if `out_p` is a \
     parameter);
   - globals or `static` storage visible at the call site;
   - facts about memory the parameters point into (`s[0..n-1]`, \
     `*p initialized`, `s null-terminated`).
   FORBIDDEN: any identifier introduced by `int foo`, `char *p = s`, \
`for (int i = ...)`, etc. inside the body. Loop counters, walking pointers, \
clamp-locals — none of these may appear in `requires` or `ensures`.
   Concrete examples of common wrong outputs:
   - WRONG: function `f(const char *s)` whose body walks `p = s; p++` \
     publishing `requires: (long)(p - s) <= INT_MAX` — `p` is a body-local \
     walking pointer; the caller has no `p`.
     RIGHT: restate in caller terms (e.g. `strlen-of-s-as-cstring <= \
     INT_MAX`), or drop the clause and let the verify pass handle the \
     body cast.
   - WRONG: a no-argument function whose body declares `int length = ...; \
     if (length < 1) length = 1;` publishing `requires: length >= 1` — \
     `length` is body-local and the function takes no parameters.
     RIGHT: omit the clause. A no-arg function CANNOT have a meaningful \
     `requires` referencing body state; that invariant belongs in `notes` \
     (or in `ensures` if it escapes via `result`).
   If you cannot restate a callee's published clause in YOUR caller-visible \
terms, drop it and rely on the verify pass instead of forwarding a \
malformed clause upward.

3. **Callee discharge is VERBATIM.** When a callee K's contract says \
`requires[P]: phi`, you may either:
   (a) DISCHARGE phi at the callsite — cite a fact already on the path \
       (e.g., "p was just assigned `malloc(n)` whose `ensures` includes \
       `result != NULL`");
   (b) PROPAGATE phi as your own `requires[P]` — verbatim (after the \
       caller-name substitution required by rule 2), optionally with the \
       callsite's path-condition prepended (e.g., `cond ==> phi`).
   You MAY NOT INVENT preconditions a callee did not declare. If callee K's \
contract is `requires[P]: true`, you must NOT add a precondition on K's \
arguments "because K probably needs them valid". If K's contract has no \
preconditions, K's arguments are unconstrained from K's perspective.

4. **NO VERDICT.** You output only `requires` / `ensures` / `modifies`. \
Whether the body satisfies its own contract is checked separately and is \
not your concern here. Never write "this function is safe/unsafe".

5. **Conservative on uncertainty**:
   - `requires`: include the obligation when in doubt (FP > FN for soundness).
   - `ensures`: weaken to a safe over-approximation (often empty / `true`).
   - `modifies`: include the location if it might be written to.

6. **Approximate when predicates grow unwieldy.** Precision is a goal, but a
   ten-clause disjunction or a deeply nested conditional defeats downstream
   reasoning. When a predicate becomes too complex, approximate
   CONSERVATIVELY:
   - `ensures`: WEAKEN (over-approximate the post-state; e.g., replace
     `result == c1 || ... || result == c20` with `result >= -2147483648 &&
     result <= 259`). Never strengthen.
   - `requires`: STRENGTHEN (under-approximate the pre-state; demand more of
     the caller to keep sound). Never weaken.
   You decide when to approximate; flag in `notes` when you do.

7. **`modifies` scope — stack + heap, not globals.** C zero-initializes
   statics and globals at program start (C11 §6.7.9¶10), so reading them
   before any explicit write is well-defined. We track `modifies` to support
   use-before-initialization reasoning at the CALLER, which only matters for:
     - **Stack locals** (uninitialized at entry; UB on read-before-write);
     - **Heap memory from `malloc` / `realloc`** (uninitialized; UB on
       read-before-write). `calloc` returns zeroed memory — exempt;
     - **Out-parameters** the caller passes in (caller treats them as
       freshly-initialized after the call returns).
   Do NOT list globals or `static` storage in `modifies` — those don't
   create use-before-init obligations.

8. **External / harness functions are summarised for you.** When a callsite \
calls a function whose body you can't see (stdlib, intrinsics, harness \
helpers), its `requires` / `ensures` / `modifies` are inlined as `// >>>` \
comments above the call. Trust those exactly; do not re-derive their \
behaviour from naming conventions or harness assumptions.

9. **Noreturn callees cut the path.** A callsite annotated `// >>> noreturn: \
true` (or whose callee block shows `noreturn: true`) does not return. If it \
sits in the THEN-arm of `if (G) noreturnCallee();`, code after the `if` \
runs only when the call did not happen — so subsequent code may assume \
`!G`. If the function's body unconditionally aborts/exits/longjmps on every \
path, set `noreturn: true` in your output (it is a property-independent \
signal — emit the same value in every per-property call).
"""


MEMSAFE_PROMPT = """\
Analyze function `{name}` for MEMORY SAFETY only.

In-scope: null deref, out-of-bounds access, use-after-free, double-free, \
init-before-read.

## Hard scope for `requires` and `ensures` in this pass

Every `requires` and `ensures` item MUST be about pointer validity, \
buffer bounds, init state, or alloc/free state — predicates like \
`p != NULL`, `0 <= i && i < n`, `*p initialized`, `not freed(p)`. \
Do NOT emit integer-overflow predicates (`x != INT_MIN`, `n <= INT_MAX/2`) \
or memory-leak predicates here — those belong to other passes.

## Method — what to publish

`requires` and `ensures` are two halves of the same walk. For each
operation in the body, ask both questions: does it need something from
the caller (UB-on-input → `requires`) AND does it establish something the
caller can rely on (size↔buffer pairing, init range, null terminator →
`ensures`).

### Caller obligation tracking (publish what the body needs in `requires`)

For each operation the body performs that could UB on its inputs, publish
the obligation in `requires`. Walk the body and check:

- **Unguarded pointer dereference**: if the body executes `*p`, `p->f`,
  or `p[i]` on parameter `p` (or a value derived from one) without first
  testing `p != NULL` on the same path, publish `p != NULL` in
  `requires`. This applies even when CIL / sv-comp encodes pointers as
  `int` — at runtime it is still a pointer and the deref still UBs.
- **Buffer access bounds**: for EVERY `buf[expr]`, `*(buf + expr)`,
  `memcpy(buf + expr, ..., len)`, or `memset(buf + expr, ..., len)` in
  the body, decompose `expr` into its constituent variables and emit
  the bound as a C expression in `requires`.
  Procedure: (1) identify the buffer pointer and its allocation size
  (parameter, struct field, or callee ensures); (2) identify the
  maximum index expression used; (3) emit `max_index < alloc_size`
  (or `offset + len <= alloc_size` for memcpy/memset).
  When the pointer, index, and size are struct fields of the same
  parameter, the bound is a relational invariant — e.g. if the body
  accesses `ctx->buf[ctx->pos + ctx->len]` and the buffer has
  `ctx->capacity` elements, emit `ctx->pos + ctx->len < ctx->capacity`.
  **Size positivity**: when the body accesses `buf[0]` or calls
  `memset(buf, 0, n * sizeof(T))` without guarding `n > 0`, publish
  `n > 0` — unsigned underflow of `n - 1` wraps to a huge value.
- **Read of uninitialized memory**: if the body READS `*p` (RHS of an
  assignment, branch test, index, arithmetic operand) for a
  non-allocator parameter `p`, publish `*p initialized` in `requires`.
  Write-only deref like `p->f = v` does NOT obligate `*p initialized` —
  only `p != NULL`. The init obligation attaches to reads, not accesses.

The rule "do NOT invent preconditions a callee did not declare" (system
rule 3) applies to **callee-forwarded** preconditions. It does NOT block
you from publishing obligations rooted in operations THIS function
performs directly on its parameters — those are required, not invented.

### Caller-observable post-state (publish what the body establishes in `ensures`)

Memsafe verification across function boundaries depends on the producer
publishing the facts the consumer needs — size↔buffer pairings, freed
state, init range, non-null returns, struct-field updates. Per system
rule 1, prose is acceptable as long as the fact is specific. Walk the
body and publish in `ensures` every concrete fact the body establishes
about returned, out-parameter, or otherwise caller-visible state:

- **Allocation size pairing**: when you `malloc(N)` (or any allocator) and
  return / store the pointer, pair the pointer with N by name in `ensures`:
  `result allocated for N bytes`, or `result points to N elements of T`.
  Always cite N. Bare "result is heap-allocated" is too weak — the
  consumer can't bound any index.
- **Element count vs byte count**: `malloc(N * sizeof(T))` cast to `T*` →
  `N elements`; `malloc(N)` cast to `T*` → `N / sizeof(T)` elements (often
  a bug). Publish the unit you actually have.
- **Non-null result from null-check-then-noreturn**: when the body has
  `p = malloc(n); if (!p) abort();` (or any noreturn callee on the null
  path — `exit`, `__assert_fail`, summarized `noreturn: true`), publish
  `result != NULL` (or `*out_p != NULL` for an out-param). The caller
  skips a redundant null-check and downstream deref obligations are
  discharged automatically.
- **Freed state**: when the body executes `free(p)`, calls a deallocator
  (`munmap`, `fclose`, `closedir`, etc.), or invokes a callee whose
  ensures publish `freed(q)`, publish `freed(p)` (or `freed(ctx->buf)`,
  `freed(*out)`) in `ensures`. Symmetric to the `not freed(p)` requires
  form — without this, cross-function use-after-free / double-free
  reasoning at the caller is blind.
- **C-string null terminator**: when you write `s[k] = '\\0'`, publish the
  terminator position: `s[k] == '\\0'`, or `s null-terminated at index k`.
  This makes the C-string LENGTH derivable as `k`. Without it, a consumer
  like `cstrlen(s)` cannot show that its loop terminates within the
  allocation. If the terminator position depends on a parameter (e.g.
  `s[length-1] = '\\0'`), name that parameter.
- **Initialization range**: when a loop writes `s[0..k]`, publish
  `s[0..k] initialized` (prose is fine here — the C-expression form would
  be a quantifier). If you only initialize a SUBSET of the buffer, publish
  the subset, not the whole buffer.
- **Out-parameter / struct-field writes**: writes via a parameter
  (`*out = v`, `ctx->data = malloc(n)`, `s->len = n`) are caller-visible
  state changes — treat them the same as writes to `result`. Publish the
  post-condition (`*out initialized`, `ctx->data allocated for n bytes`,
  `s->len == n`). The size-pairing / null-terminator / init-range
  bullets above all apply when the target is `*out` or a struct field
  reachable from a parameter.
- **Callee post-state propagation**: if a callsite's callee.ensures
  publish a fact about state still observable at function exit (callee
  inits `*out` and you return without re-touching it; callee allocates
  and you return its result; callee frees `*p` and you don't reallocate),
  propagate the callee's ensures as your own. Don't drop callee facts
  just because this function didn't establish them directly.
- **Pre-existing facts**: anything you LEARNED from a callsite (e.g.
  `t = malloc(...); if (t == 0) myexit(1);` → `t != NULL` after the
  branch) and that is still true at the function exit is also publishable
  if a caller would care.

If a producer fails to publish one of these facts, the consumer either
(a) can't write a meaningful `requires` (FN — bug missed because the
relevant invariant has no name), or (b) writes a `requires` the caller
cannot discharge (FP — false alarm propagated up the chain). Both are
worse than a slightly verbose `ensures`.

### When a fact is path-conditional

Per system rule 6, `ensures` may always WEAKEN. If the fact (e.g.
`freed(p)`) holds only on some paths and the guard isn't expressible in
caller-observable terms (it depends on a body-local variable), pick ONE
of:
  (a) publish unconditionally if the worst-case interpretation is sound
      for the caller (e.g. `freed(p)` covers both "always freed" and
      "may have freed" — caller treats freed-or-not as freed);
  (b) drop the clause (sound default for `ensures`).
NEVER fabricate a body-local name in the condition just to keep the
clause. The verify pass has the body and reasons about paths directly.

The header block lists each callee's published pre/post for memsafe. Use them \
verbatim; do NOT invent preconditions a callee did not declare.

{callee_block}

=== SOURCE ===
{source}

Output JSON exactly matching:
{{
  "requires": ["<C boolean expression>", ...],
  "ensures":  ["<C boolean expression or short prose>", ...],
  "modifies": ["<sym>", ...],
  "notes":    "<one-line context propagated to YOUR caller alongside contract>"
}}

## Format rules

**`requires` MUST be C boolean expressions** that a caller can evaluate
at the callsite — `p != NULL`, `n > 0`, `i + k < ctx->size`. Avoid prose
("buffer must be large enough"); write the actual bound instead.
Exception: `*p initialized` is acceptable because C has no syntax for it.

**`ensures`** may use prose for facts C cannot express (init ranges,
allocation pairing), but prefer C expressions when possible:
`result != NULL`, `*out == 0`, `ctx->len == n`.

Guidance:
- `requires` examples: `ctx != NULL`, `n > 0`,
  `ctx->pos + ctx->len <= ctx->capacity`.
- `ensures` examples: `result allocated for n * sizeof(T) bytes`,
  `result != NULL`, `s->buf[0..n-1] initialized`, `freed(p)`.
- `modifies`: list stack locals and heap regions written here (out-params,
  *p where p was malloc'd in this function, etc.). SKIP globals/statics
  (zero-init at startup → no use-before-init obligation).
- If function has no memsafe obligations and no memsafe-relevant ensures,
  output empty arrays. Empty is the right answer for many simple functions.
"""


MEMLEAK_PROMPT = """\
Analyze function `{name}` for MEMORY LEAKS only.

In-scope: every heap allocation either (a) released within the function on \
every return path, (b) returned to the caller, or (c) stored in a \
caller-visible location. Anything else leaks.

## Hard scope for `requires` and `ensures` in this pass

Every `requires` and `ensures` item MUST be about resource ownership and \
release — clauses like `caller releases <sym>`, `acquires fd: caller \
must close`, `all acquisitions released`, `<callee>.requires[memleak] \
holds`. Do NOT emit integer-overflow or pointer-validity predicates here.

{callee_block}

=== SOURCE ===
{source}

Output JSON exactly matching:
{{
  "requires": ["<expr or short clause>", ...],
  "ensures":  ["<expr or short clause>", ...],
  "modifies": ["<sym>", ...],
  "notes":    "<one-line context propagated to YOUR caller alongside contract>"
}}

Guidance:
- `requires` examples: `caller releases <sym>` (function acquires resource and
  hands it off), `<callee>.requires[memleak] holds` (propagated verbatim).
- `ensures` examples: `no resource acquired`, `acquires fd: caller must close`,
  `all acquisitions released`.
- If the function does not allocate, free, or call anything that does, output
  empty arrays.
"""


OVERFLOW_PROMPT = """\
Analyze function `{name}` for INTEGER UB. SUMMARY ONLY — produce \
preconditions/postconditions that downstream verification can use. \
DO NOT decide whether a bug exists (that's the verify pass).

## Hard scope for `requires` and `ensures` in this pass

Every `requires` and `ensures` item MUST be about integer values:
signed-overflow ranges (`x != INT_MIN`, `n <= INT_MAX/2`), divisor
nonzero (`d != 0`), shift bounds (`0 <= s && s < 32`), or value-range
results (`result >= 0 && result <= INT_MAX`).

Do NOT emit pointer-validity predicates (`p != NULL`, `p != NP`),
initialization predicates (`*p initialized`, `field set`), or other
non-integer concerns here. Those belong to the memsafe pass and will
appear in this function's other property entries — leave them out.

## In-scope operations

- Signed arithmetic: `+`, `-`, `*`, `++`, `--`, unary `-` (overflow risk).
- Division / modulo: `/`, `%` (zero-divisor, plus `INT_MIN/-1`, `INT_MIN%-1`).
- Shifts: `<<`, `>>` (negative amount, amount >= bit-width of promoted left
  operand, left-shifting a negative signed value).

## Method — value-range analysis

Walk the function statement by statement, tracking a value range per integer
variable. Emit each range either as a code predicate (`x >= 1 && x <= 100`)
or as a compact value-range form (`x: [1, 100]`, `n: [INT_MIN+1, INT_MAX]`).
Pick whichever is shorter for the case at hand. Source language is whatever
the input file uses (C or C++); use that language's expression syntax.

1. **Initialise**: from C constants, callee `ensures[overflow]` ranges, and
   any callee `requires[overflow]` already discharged on this path.
2. **Narrow on branches**: after `if (x > 0)`, treat x as `x >= 1` on the true
   side and `x <= 0` on the false side. After a callsite whose summary
   says it aborts unless `c` (a callee `ensures` like `c on returning path`),
   treat `c` as a path fact for subsequent code.
3. **Skip dead branches**: if path facts make a branch's guard unsatisfiable
   (e.g. a `case` selector that earlier code has constrained out, or an
   `if (false)` derivable from prior assignments), treat that branch's body
   as unreachable. Do NOT publish `requires` for operations on paths that
   cannot be entered.
4. **Arithmetic**: combine the operand ranges. If the result range can fall
   outside the operand type's range under the inputs you've tracked, emit a
   `requires` predicate that excludes the bad inputs.
5. **Callee discharge** (verbatim): for each callee K with
   `requires[overflow]: phi`, EITHER discharge phi at the callsite by citing
   the path facts that imply it, OR propagate phi as YOUR `requires`
   (verbatim, optionally with the path-condition prepended:
   `cond ==> phi`). NEVER invent a precondition a callee did not declare.

## Output form — code or value-range, not English

- `requires` items are boolean expressions or value-range forms that must
  hold on entry. Examples:
    `n != INT_MIN`                         (function computes `-n` on int)
    `n: [INT_MIN+1, INT_MAX]`              (same fact, range form)
    `n <= INT_MAX / 2`                     (function computes `2*n` on int)
    `divisor != 0`                         (function computes `x / divisor`)
    `shift_amt: [0, 31]`                   (function does `x << shift_amt`)
    `cond ==> n != INT_MIN`                (precondition only on a path)

- `ensures` items are boolean expressions or value-range forms about the
  return value, out-parameters, and values written to struct fields via
  pointer parameters on exit. Use the literal name `result` for the return
  value, `*out_p` for a simple out-parameter, and `p->field` for a struct
  field written through a pointer parameter. Examples:
    `result: [0, INT_MAX]`                 (return value range)
    `result == a + b`                      (exact value, holds under requires)
    `*out_p >= 0`                          (out-parameter postcondition)
    `result != 0 ==> *err == 0`            (conditional postcondition)
    `p->depth: [0, MAX_DEPTH]`             (struct field written via pointer param)

  Only report when the bound is derivable from the body and narrower than
  the full type range — skip tautological ranges like `int: [INT_MIN, INT_MAX]`.

  If the function does not return an integer or has no integer effect,
  `ensures` may be empty. Empty is a valid answer.

- `modifies` items are stack locals and heap locations whose value the
  function changes (so the caller can reason about use-before-init for
  out-params and freshly-allocated memory). SKIP globals and `static`
  storage — C zero-inits them, so they create no use-before-init
  obligation.

- `notes` is one line of free-form context shown to YOUR caller alongside
  your contract (and to the verifier). Useful for facts that don't fit
  cleanly in `requires` / `ensures` (e.g. "this function aborts unless cond
  holds", "loop initialises s[0..length-1]"). Keep it short.

## Reminders for compositional discipline

- A function with no signed arithmetic, no division/modulo, no shifts, AND
  whose callees all publish empty `requires[overflow]` and `ensures[overflow]`
  → output empty arrays.
- **Mitigate external inputs.** Any signed op whose operand comes from an
  external source (parameter, struct field via pointer, global) and whose
  full type range admits UB MUST produce a `requires` constraining the
  input to the safe sub-range. Do NOT skip because "the value is probably
  small" — emit the constraint and let the verify pass judge feasibility.
  Examples:
    `int x = (int)p->count;  x - 1` → requires: `p->count` (unsigned) `<= INT_MAX`
    `a << n` where n is a signed param → requires: `n: [0, 31]`
- Each `requires` must be a concrete predicate that ACTUALLY excludes the
  UB case (`x != INT_MIN`, `denom != 0`, `0 <= s < 32`).
- **No tautologies.** `x: [INT_MIN, INT_MAX]` for an `int` is the full
  range — it excludes nothing. Drop it.
- **No memsafe concerns.** Do not emit pointer-validity, buffer-size, or
  availability constraints here (e.g. `avail_in >= 5`). Those belong to
  the memsafe pass.
- **Overapproximate complex expressions.** If the exact safe range depends
  on runtime state (array lookups, loop variables, multi-field invariants),
  widen to a simpler bound on the entry-visible inputs. For example, if
  safety requires `bi_valid + tree[sym].len <= 16` and `tree[sym].len` is
  at most 15, emit `bi_valid: [0, 1]` or `bi_valid + 15 <= 16` — not a
  constraint that references `tree[sym]`.
- **Shift operands.** `1U << n` or `x << n` is UB when `n >= bit-width`
  of the promoted left operand. Always emit `n: [0, W-1]` (W = 32 or 64)
  for shift amounts read from external sources.
- **Published output bounds.** When the function writes a value to a struct
  field (or any caller-visible location) and that value's range is derivable
  and narrower than the type, publish it in `ensures`. Callers reading that
  field in a shift or comparison cannot derive the bound themselves.
  Example: function assigns `p->count = n` where `n < MAX` → `ensures: p->count: [0, MAX-1]`.

## Resolve typedefs

The source may be preceded by a `## Type definitions` block listing
`typedef` and `struct` definitions. Before reasoning about any
variable's signedness or bit-width, resolve its declared type through
those typedefs to the underlying primitive. An op on an unsigned type
is NOT signed overflow.

## C-semantics reminders (apply BEFORE flagging an op as UB)

- **Integer promotion.** Operands narrower than `int` (`char`, `short`,
  `_Bool`) are promoted to `int` before arithmetic; check overflow at the
  PROMOTED type, not the original.
- **Unsigned wrap is well-defined.** `unsigned` arithmetic wraps modulo
  2^N — not UB. Do not flag wrap on unsigned types.
- **Literal type follows C rules.** Unsuffixed decimal constants take the
  smallest of `int` / `long` / `long long` that fits; `LL` / `ULL` force
  long-long. The leading `-` on a literal is the unary-minus operator
  applied AFTER the literal's type is fixed.
- **Compare against the result type's range.** A signed op overflows only
  when its mathematical result falls outside the result type's
  representable range. Values that fit exactly (including the type's
  min/max) are not overflow. Worked example: `65536 * -32768 ==
  -2147483648 == INT_MIN` — the product LANDS ON the boundary, it is
  representable, NOT overflow. Check the boundary explicitly before
  flagging a multiplication that produces the type's min/max.
- **Unsigned-to-signed within-range cast.** `(int) u` where `u` fits in
  `int` (i.e. `u <= INT_MAX`) is well-defined. Conversions from unsigned
  to signed are only problematic when the value exceeds the signed
  target's range, and even then modern compilers wrap rather than UB.
  Don't flag `int n = (int) strlen(s);` when `s` is short enough that
  `strlen(s) <= INT_MAX` is the obvious case.

## Data model

{data_model_note}

The header block lists each callee's published pre/post for overflow. Use
them verbatim; do NOT invent preconditions a callee did not declare.

{callee_block}

=== SOURCE ===
{source}

Output JSON exactly matching:
{{
  "requires": ["<expr or x: [lo, hi]>", ...],
  "ensures":  ["<expr or x: [lo, hi]>", ...],
  "modifies": ["<sym>", ...],
  "notes":    "<one-line context propagated to YOUR caller alongside contract>"
}}
"""


PROPERTY_PROMPT: dict[str, str] = {
    "memsafe": MEMSAFE_PROMPT,
    "memleak": MEMLEAK_PROMPT,
    "overflow": OVERFLOW_PROMPT,
}


# ── Verify-pass prompts (lifted from contract_pipeline.py:1036+, with
#    de-sv-comp edit on the system prompt rule 5).

VERIFY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "analysis": {"type": "string"},
                    "is_ub": {"type": "boolean"},
                },
                "required": ["kind", "analysis", "is_ub"],
            },
        },
    },
    "required": ["issues"],
}


VERIFY_SYSTEM_PROMPT = """\
You verify whether a C function body satisfies its own published Hoare-style
contract for a single safety property. You are NOT producing the contract
(that was a prior pass) — you are looking for property violations *inside*
the body, given the body, the contract, and each callee's contract.

## Hard rules

1. **Assume `requires` hold.** The function's published `requires[P]` are
   given to you as facts on entry. Do NOT report a "missing precondition"
   or recommend tightening the contract. If a precondition you wish existed
   is not present, the body must establish it before the relevant operation
   or that operation is a violation.

2. **Discharge callee `requires[P]` at every callsite.** Each callsite is
   annotated with `// >>> callee contract for P` showing requires/ensures/
   modifies. If a callee's `requires[P]` may NOT hold at the callsite given
   the path facts, that is a violation in THIS function (the caller failed
   to discharge it).

3. **Stay inside the property.** Only flag issues belonging to property P —
   memsafe, memleak, or overflow as instructed. Do NOT mix.

4. **Be specific.** Each issue cites a concrete operation, not a category
   ("the function does arithmetic so it might overflow" is not an issue —
   "`x = 2147483647 + 1` overflows on the unconditional path" is).

5. **External / harness functions are summarised for you.** Each callsite is
   annotated with the callee's `requires` / `ensures` / `modifies`. Trust
   those exactly — including for stdlib (`malloc`, `free`, etc.) and
   noreturn helpers (declared with `__attribute__((noreturn))`, or
   summarized as `noreturn: true`). Don't re-derive their behaviour from
   the name.

6. **Noreturn callees cut the path.** A callsite annotated
   `// >>> noreturn: true` does not return. Code after such a call (on the
   same straight-line path) is unreachable. If the call sits in the
   THEN-arm of `if (G) noreturnCallee();`, code after the `if` runs only
   when the call did not happen — so subsequent code may assume `!G`.

7. **Empty list is the right answer when the body is safe under its
   published contract.**

8. **Global / static objects are never NULL.** Variables annotated
   `// global` in the type-definitions section have non-NULL addresses.
"""


VERIFY_MEMSAFE_PROMPT = """\
VERIFY function `{name}` for MEMORY SAFETY violations under its published
memsafe contract.

In-scope kinds (use these exact `kind` strings):
  - `null_deref`         — `*p`, `p->f`, `p[i]` with p potentially NULL
  - `buffer_overflow`    — `a[i]` or `*(p + offset)` with index/offset
                           potentially outside [0, len). Include cases where
                           integer overflow or type confusion in the offset
                           computation leads to an out-of-bounds pointer.
  - `use_after_free`     — deref of a pointer freed earlier on the path
  - `double_free`        — free of a pointer already freed on the path
  - `invalid_free`       — free of a pointer not from malloc / not at base
  - `uninitialized_use`  — read of a stack/heap byte not yet written
  - `callee_requires`    — a callsite where callee.requires[memsafe] may not
                           hold given the path facts (cite the callee + which
                           clause); this is the caller's bug to flag

## Function's published memsafe contract (assume `requires` hold on entry)
{own_contract}

## Unsigned wrap in bounds guards

Unsigned arithmetic wraps modulo 2^N — this is not integer UB, but when a
wrapped result is used in a comparison that guards a memory operation
(e.g. `if (pos + len >= space)` before `memcpy(buf + pos, src, len)`),
the wrap can make the guard evaluate false, skipping a realloc or bounds
check. Flag as `buffer_overflow` when: (1) unsigned addition or
multiplication in a branch condition can wrap on the target data model,
AND (2) the unguarded (false) path reaches a memory access that assumes
the check passed. Pay special attention to `size_t` sums that feed
allocation-size or bounds comparisons — on ILP32, `pos + len` can wrap
past SIZE_MAX even when both operands are individually valid.

## Frontend warnings

If the source begins with a `// FRONTEND WARNINGS (clang -Wall) ...` block, \
treat each warning as a candidate issue. Decide if it is feasible at runtime \
(reachable path, operand range admits UB) and emit the matching `issues[]` \
entry. Skip warnings that are clearly dead code or already discharged by the \
contract.

{callee_block}

=== SOURCE (callsites annotated with `// >>> callee contract for memsafe`) ===
{source}

Output JSON exactly matching:
{{
  "issues": [
    {{"kind": "<one of the kinds above>",
     "line": <int|null>,
     "analysis": "<one sentence: cite the op, state whether path facts \
discharge it, conclude safe or unsafe>",
     "is_ub": true}}
  ]
}}

Set `is_ub` to `false` when analysis shows the op is safe. \
Only `is_ub: true` entries become real issues.
"""


VERIFY_MEMLEAK_PROMPT = """\
VERIFY function `{name}` for MEMORY LEAKS under its published memleak contract.

In-scope kinds:
  - `memory_leak`     — heap allocation on a path that doesn't reach `free`,
                        isn't returned to the caller, and isn't stored in
                        caller-visible memory before the function returns
  - `callee_requires` — a callsite where callee.requires[memleak] may not hold
                        (e.g. the caller fails to release a resource the
                        callee declares the caller must release)

## Function's published memleak contract (assume `requires` hold on entry)
{own_contract}

## Frontend warnings

If the source begins with a `// FRONTEND WARNINGS (clang -Wall) ...` block, \
treat each warning as a candidate `memory_leak` (or `callee_requires`) when it \
implies an unreleased acquisition; otherwise leave it alone.

{callee_block}

=== SOURCE (callsites annotated with `// >>> callee contract for memleak`) ===
{source}

Output JSON exactly matching:
{{
  "issues": [
    {{"kind": "memory_leak|callee_requires",
     "line": <int|null>,
     "analysis": "<one sentence: cite the allocation/callsite, state whether \
it leaks or is released, conclude safe or unsafe>",
     "is_ub": true}}
  ]
}}

Set `is_ub` to `false` when analysis shows no leak. \
Only `is_ub: true` entries become real issues.
"""


VERIFY_OVERFLOW_PROMPT = """\
VERIFY function `{name}` for INTEGER UB under its published overflow contract.

In-scope kinds:
  - `integer_overflow` — signed `+` / `-` / `*` / `++` / `--` / unary `-`
                         whose result can fall outside the operand type
  - `division_by_zero` — `/` or `%` with a divisor that may be zero
  - `shift_ub`         — `<<` / `>>` with negative amount, amount >= bit-width
                         of the promoted left operand, or left-shift of a
                         negative signed value
  - `callee_requires`  — a callsite where callee.requires[overflow] may not
                         hold (cite which clause)

NOT in scope (handled by the memsafe pass):
  - Pointer arithmetic (`p + i`, `p - q`). Whether `p + i` stays within
    the allocation is a buffer-bounds question, not integer overflow.
    Only flag the integer sub-expression if it overflows on its OWN type
    before being added to the pointer.

## Method (value-range; use the source language's expression syntax)

Walk the body statement by statement, tracking each integer's range from:
  (a) the function's published `requires[overflow]` — these are FACTS;
      propagate them to every variable derived from the constrained input,
  (b) C constants assigned along the path,
  (c) callee `ensures[overflow]` after each callsite,
  (d) branch narrowing (`if (x > 0)` ⇒ `x >= 1` on the true side),
  (e) `assume_abort_if_not(c)` adds `c` to the path.
For each in-scope op, ask: can the operand range admit UB given the
narrowed ranges? If yes, that op is an issue. If the requires already
exclude the dangerous sub-range, the op is safe — do not re-flag it.

**Skip dead branches.** If path facts make a branch's guard unsatisfiable
(e.g. a `case` selector ruled out by an earlier `if`, code after a
noreturn callsite on the same straight-line path), do NOT flag operations
inside that branch. Unreachable UB is not UB.

For each callsite: do the path facts imply every clause of
callee.requires[overflow]? If not, emit `callee_requires`.

## C-semantics reminders (apply BEFORE flagging an op as UB)

- **Integer promotion.** Operands narrower than `int` (`char`, `short`,
  `_Bool`) are promoted to `int` before arithmetic; check overflow at the
  PROMOTED type, not the original.
- **Unsigned wrap is well-defined.** `unsigned` arithmetic wraps modulo
  2^N — not UB. Do not flag wrap on unsigned types.
- **Literal type follows C rules.** Unsuffixed decimal constants take the
  smallest of `int` / `long` / `long long` that fits; `LL` / `ULL` force
  long-long. The leading `-` on a literal is the unary-minus operator
  applied AFTER the literal's type is fixed.
- **Compare against the result type's range.** A signed op overflows only
  when its mathematical result falls outside the result type's
  representable range. Values that fit exactly (including the type's
  min/max) are not overflow. Worked example: `65536 * -32768 ==
  -2147483648 == INT_MIN` — product LANDS ON the boundary, representable,
  NOT overflow.
- **Unsigned-to-signed within-range cast.** `(int) u` where `u` fits in
  `int` is well-defined; only flag conversion UB when the unsigned value
  exceeds the signed target's range.
- **Noreturn callees in guards.** A callsite annotated
  `// >>> noreturn: true` (e.g. `abort`, `exit`) does not return; if it
  sits in `if (G) noreturnCallee();`, code after the `if` may assume `!G`.

## Resolve typedefs

The source may be preceded by a `## Type definitions` block listing
`typedef` and `struct` definitions. Before reasoning about any
variable's signedness or bit-width, resolve its declared type through
those typedefs to the underlying primitive. An op on an unsigned type
is NOT signed overflow.

## Data model

{data_model_note}

## Function's published overflow contract (assume `requires` hold on entry)
{own_contract}

## Frontend warnings

If the source begins with a `// FRONTEND WARNINGS (clang -Wall) ...` block, \
treat each warning as a candidate `integer_overflow` / `division_by_zero` / \
`shift_ub` issue. Constant-folded expressions whose result clang flagged \
(e.g. `INT_MAX + 1`) are STILL UB at the source level — emit the issue even \
when no `add` instruction survives in IR. Skip warnings only when the path is \
provably unreachable.

{callee_block}

=== SOURCE (callsites annotated with `// >>> callee contract for overflow`) ===
{source}

Output JSON exactly matching:
{{
  "issues": [
    {{"kind": "integer_overflow|division_by_zero|shift_ub|callee_requires",
     "line": <int|null>,
     "analysis": "<one sentence: cite the op, state operand type and \
range, conclude UB or well-defined>",
     "is_ub": true}}
  ]
}}

Set `is_ub` to `false` for ops that are well-defined (unsigned wrap, \
guarded paths, values in range). Only `is_ub: true` entries become \
real issues.
"""


VERIFY_PROMPT: dict[str, str] = {
    "memsafe": VERIFY_MEMSAFE_PROMPT,
    "memleak": VERIFY_MEMLEAK_PROMPT,
    "overflow": VERIFY_OVERFLOW_PROMPT,
}


_DATA_MODEL_NOTES: dict[str, str] = {
    "LP64": ("LP64 (sizeof: int=4, long=8, long long=8, void*=8). "
             "Type ranges: int [-2^31, 2^31-1]; long/long long [-2^63, 2^63-1]."),
    "ILP32": ("ILP32 (sizeof: int=4, long=4, long long=8, void*=4). "
              "Type ranges: int / long [-2^31, 2^31-1]; long long [-2^63, 2^63-1]."),
}


def data_model_note(model: str | None) -> str:
    """Render a one-paragraph data-model note for the prompt. Defaults to
    LP64 when the task didn't declare one."""
    key = (model or "LP64").upper()
    return _DATA_MODEL_NOTES.get(key, _DATA_MODEL_NOTES["LP64"])
