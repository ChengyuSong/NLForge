"""Stdlib / harness contract seeds for opaque callees.

Lifted from `scripts/contract_pipeline.py:142-251`, then split:
  - This module: pure libc / POSIX / compiler-builtin contracts. Always
    loaded by the pipeline.
  - `svcomp_stdlib.py`: sv-comp helpers (every `__VERIFIER_*`,
    `assume_abort_if_not`). Loaded only when running on sv-comp benchmarks.

Pre-built `CodeContractSummary` entries are seeded into the `summaries`
dict before the topo walk so the existing callee-inlining flow surfaces
them at every callsite — no code path needs to special-case "external
function".
"""

from __future__ import annotations

from typing import Any

from .models import CodeContractSummary


def _summary(
    name: str,
    *,
    noreturn: bool = False,
    **per_prop: Any,
) -> CodeContractSummary:
    """Compact constructor: _summary("malloc", memsafe={"ensures": [...]})."""
    s = CodeContractSummary(function=name, properties=list(per_prop.keys()))
    for prop, slots in per_prop.items():
        s.requires[prop] = list(slots.get("requires", []))
        s.ensures[prop]  = list(slots.get("ensures",  []))
        s.modifies[prop] = list(slots.get("modifies", []))
        n = slots.get("notes", "")
        if isinstance(n, str) and n:
            s.notes[prop] = n
    s.noreturn = noreturn
    return s


def _build_libc_contracts() -> dict[str, CodeContractSummary]:
    out: dict[str, CodeContractSummary] = {}

    # ── Allocators ──
    out["malloc"] = _summary(
        "malloc",
        memsafe={"ensures": ["result is NULL or allocated for size bytes (uninitialized)"]},
        memleak={"ensures": ["acquires heap allocation: caller must release result if non-NULL"]},
    )
    out["calloc"] = _summary(
        "calloc",
        memsafe={"ensures": ["result is NULL or allocated for nmemb*size bytes",
                             "result[0..nmemb*size-1] initialized to zero when result != NULL"]},
        memleak={"ensures": ["acquires heap allocation: caller must release result if non-NULL"]},
    )
    out["realloc"] = _summary(
        "realloc",
        memsafe={
            "requires": ["ptr is NULL or previously allocated and not yet freed"],
            "ensures":  [
                "result is NULL or allocated for size bytes",
                "ptr ownership transferred to result on success "
                "(do not use ptr after)",
            ],
        },
        memleak={"ensures": [
            "on success: acquires heap allocation tied to result; releases ptr"
        ]},
    )
    out["reallocarray"] = _summary(
        "reallocarray",
        memsafe={"requires": ["ptr is NULL or previously allocated and not yet freed"],
                 "ensures":  ["result is NULL or allocated for nmemb*size bytes",
                              "ptr ownership transferred to result on success"]},
        memleak={"ensures": ["on success: acquires heap allocation tied to result; releases ptr"]},
    )

    out["aligned_alloc"] = _summary(
        "aligned_alloc",
        memsafe={"requires": ["alignment is a power of 2",
                              "size is a multiple of alignment"],
                 "ensures":  ["result == NULL || allocated(result, size)"]},
        memleak={"ensures": ["acquires heap allocation: caller must free(result) if non-NULL"]},
    )
    out["asprintf"] = _summary(
        "asprintf",
        memsafe={"requires": ["strp != NULL",
                              "fmt != NULL && fmt is NUL-terminated"],
                 "ensures":  ["result == -1 || (*strp != NULL "
                              "&& *strp is NUL-terminated)"]},
        memleak={"ensures": ["on success: acquires heap allocation via *strp; "
                             "caller must free(*strp)"]},
    )

    # ── Stack alloc (no leak obligation) ──
    for name in ("alloca", "__builtin_alloca"):
        out[name] = _summary(
            name,
            memsafe={"ensures": ["result != NULL",
                                 "result allocated for size bytes on stack (uninitialized)",
                                 "result valid only until enclosing function returns"]},
        )

    # ── free ──
    out["free"] = _summary(
        "free",
        memsafe={"requires": ["ptr is NULL or from malloc/calloc/realloc and not yet freed"]},
        memleak={"ensures": ["releases ptr"]},
        # ptr's referent becomes invalid; modeled as modifying the heap region.
    )

    # ── Process termination (libc; noreturn) ──
    out["abort"] = _summary(
        "abort",
        noreturn=True,
        overflow={"ensures": ["does not return"]},
    )
    out["exit"] = _summary(
        "exit",
        noreturn=True,
        overflow={"ensures": ["does not return"]},
        memleak={"ensures":  ["all program-lifetime allocations released by OS"]},
    )
    for n in ("_exit", "_Exit", "quick_exit", "__assert_fail", "__assert",
              "__assert_perror_fail", "longjmp", "siglongjmp",
              "pthread_exit", "thrd_exit"):
        out[n] = _summary(n, noreturn=True)

    # ── errno / setjmp ──
    out["__errno_location"] = _summary(
        "__errno_location",
        memsafe={"ensures": ["result != NULL",
                             "result points to thread-local int (errno)"]},
    )
    out["_setjmp"] = _summary(
        "_setjmp",
        memsafe={"requires": ["env points to a writable jmp_buf"],
                 "ensures":  ["env may be used by a subsequent longjmp "
                              "while the enclosing function is live"]},
        # Modeled as a normal-return call; longjmp is modeled separately as noreturn.
    )

    # ── string / mem ──
    out["memcpy"] = _summary(
        "memcpy",
        memsafe={"requires": ["dest != NULL && writable(dest, n)",
                              "src != NULL && readable(src, n)",
                              "no overlap between [dest, dest+n) and [src, src+n)"],
                 "ensures":  ["initialized(dest, n)", "result == dest"]},
    )
    out["memmove"] = _summary(
        "memmove",
        memsafe={"requires": ["dest != NULL && writable(dest, n)",
                              "src != NULL && readable(src, n)"],
                 "ensures":  ["initialized(dest, n)", "result == dest"]},
    )
    out["memset"] = _summary(
        "memset",
        memsafe={"requires": ["s != NULL && writable(s, n)"],
                 "ensures":  ["initialized(s, n)", "result == s"]},
    )
    out["memcmp"] = _summary(
        "memcmp",
        memsafe={"requires": ["s1 != NULL && readable(s1, n)",
                              "s2 != NULL && readable(s2, n)"]},
    )
    out["strlen"] = _summary(
        "strlen",
        memsafe={"requires": ["s != NULL && s is NUL-terminated"]},
    )
    out["strcmp"] = _summary(
        "strcmp",
        memsafe={"requires": ["s1 != NULL && s1 is NUL-terminated",
                              "s2 != NULL && s2 is NUL-terminated"]},
    )
    out["strncmp"] = _summary(
        "strncmp",
        memsafe={"requires": ["s1 != NULL && readable(s1, n)",
                              "s2 != NULL && readable(s2, n)"]},
    )
    out["strcpy"] = _summary(
        "strcpy",
        memsafe={"requires": ["dest != NULL",
                              "src != NULL && src is NUL-terminated",
                              "writable(dest, strlen(src) + 1)",
                              "no overlap between dest and src"],
                 "ensures":  ["dest is NUL-terminated", "result == dest"]},
    )
    out["strncpy"] = _summary(
        "strncpy",
        memsafe={"requires": ["dest != NULL && writable(dest, n)",
                              "src != NULL && readable(src, n)"],
                 "ensures":  ["initialized(dest, n)", "result == dest"]},
    )
    out["strdup"] = _summary(
        "strdup",
        memsafe={"requires": ["s != NULL && s is NUL-terminated"],
                 "ensures":  ["result == NULL || (result is NUL-terminated "
                              "&& allocated(result, strlen(s) + 1))"]},
        memleak={"ensures": ["acquires heap allocation: caller must free(result) if non-NULL"]},
    )
    out["strndup"] = _summary(
        "strndup",
        memsafe={"requires": ["s != NULL && readable(s, n)"],
                 "ensures":  ["result == NULL || (result is NUL-terminated "
                              "&& allocated(result, min(n, strlen(s)) + 1))"]},
        memleak={"ensures": ["acquires heap allocation: caller must free(result) if non-NULL"]},
    )
    out["strcat"] = _summary(
        "strcat",
        memsafe={"requires": ["dest != NULL && dest is NUL-terminated",
                              "src != NULL && src is NUL-terminated",
                              "writable(dest, strlen(dest) + strlen(src) + 1)",
                              "no overlap between dest and src"],
                 "ensures":  ["dest is NUL-terminated", "result == dest"]},
    )
    out["strerror"] = _summary(
        "strerror",
        memsafe={"ensures": ["result != NULL",
                             "result points to a NUL-terminated static/thread-local "
                             "buffer; do not free; may be overwritten by next call"]},
    )

    # ── string -> number ──
    out["atof"] = _summary(
        "atof",
        memsafe={"requires": ["nptr is non-NULL and points to a NUL-terminated string"]},
    )

    # ── math (write through pointer) ──
    out["frexp"] = _summary(
        "frexp",
        memsafe={"requires": ["exp is non-NULL and writable for sizeof(int)"]},
    )
    out["modf"] = _summary(
        "modf",
        memsafe={"requires": ["iptr is non-NULL and writable for sizeof(double)"]},
    )
    out["pow"] = _summary(
        "pow",
        memsafe={"ensures": ["pure: no memory effects"]},
    )

    # ── time ──
    out["gmtime"] = _summary(
        "gmtime",
        memsafe={"requires": ["timer is non-NULL and readable for sizeof(time_t)"],
                 "ensures":  ["result is NULL on failure or points to a static struct tm",
                              "static buffer may be overwritten by subsequent "
                              "gmtime/localtime calls; do not free"]},
    )

    # ── stdio ──
    out["fopen"] = _summary(
        "fopen",
        memsafe={"requires": ["pathname != NULL && pathname is NUL-terminated",
                              "mode != NULL && mode is NUL-terminated"],
                 "ensures":  ["result == NULL || result is a valid open FILE*"]},
        memleak={"ensures": ["acquires FILE resource: caller must fclose(result) if non-NULL"]},
    )
    out["fdopen"] = _summary(
        "fdopen",
        memsafe={"requires": ["fd is an open file descriptor",
                              "mode != NULL && mode is NUL-terminated"],
                 "ensures":  ["result == NULL || result is a valid open FILE*"]},
        memleak={"ensures": ["acquires FILE resource: caller must fclose(result) if non-NULL"]},
    )
    out["fclose"] = _summary(
        "fclose",
        memsafe={"requires": ["stream != NULL && stream is a valid open FILE*"]},
        memleak={"ensures": ["releases FILE resource and underlying fd"]},
    )
    out["tmpfile"] = _summary(
        "tmpfile",
        memsafe={"ensures": ["result == NULL || result is a valid open FILE*"]},
        memleak={"ensures": ["acquires FILE resource: caller must fclose(result) if non-NULL"]},
    )
    out["fread"] = _summary(
        "fread",
        memsafe={"requires": ["ptr != NULL && writable(ptr, size * nmemb)",
                              "stream != NULL && stream is a valid open FILE*"],
                 "ensures":  ["initialized(ptr, size * result)"]},
    )
    out["fwrite"] = _summary(
        "fwrite",
        memsafe={"requires": ["ptr != NULL && readable(ptr, size * nmemb)",
                              "stream != NULL && stream is a valid open FILE*"]},
    )
    out["fprintf"] = _summary(
        "fprintf",
        memsafe={"requires": ["stream != NULL && stream is a valid open FILE*",
                              "format != NULL && format is NUL-terminated"]},
    )
    out["printf"] = _summary(
        "printf",
        memsafe={"requires": ["format != NULL && format is NUL-terminated"]},
    )
    out["fputs"] = _summary(
        "fputs",
        memsafe={"requires": ["s != NULL && s is NUL-terminated",
                              "stream != NULL && stream is a valid open FILE*"]},
    )
    out["getline"] = _summary(
        "getline",
        memsafe={"requires": ["lineptr != NULL", "*lineptr == NULL || allocated(*lineptr, *n)",
                              "n != NULL", "stream != NULL && stream is a valid open FILE*"],
                 "ensures":  ["result == -1 || (*lineptr != NULL "
                              "&& *lineptr is NUL-terminated)"]},
        memleak={"ensures": ["may realloc *lineptr: caller must free(*lineptr)"]},
    )
    out["ferror"] = _summary(
        "ferror",
        memsafe={"requires": ["stream != NULL && stream is a valid open FILE*"]},
    )
    out["fflush"] = _summary(
        "fflush",
        memsafe={"requires": ["stream == NULL || stream is a valid open FILE*"]},
    )
    out["remove"] = _summary(
        "remove",
        memsafe={"requires": ["pathname != NULL && pathname is NUL-terminated"]},
    )
    out["perror"] = _summary(
        "perror",
        memsafe={"requires": ["s == NULL || s is NUL-terminated"]},
    )
    out["unlink"] = _summary(
        "unlink",
        memsafe={"requires": ["pathname != NULL && pathname is NUL-terminated"]},
    )
    out["snprintf"] = _summary(
        "snprintf",
        memsafe={"requires": ["(s == NULL && n == 0) || (s != NULL && writable(s, n))",
                              "fmt != NULL && fmt is NUL-terminated"],
                 "ensures":  ["n > 0 && s != NULL => initialized(s, min(n, result+1)) "
                              "&& s is NUL-terminated within n bytes"]},
    )
    out["vsnprintf"] = _summary(
        "vsnprintf",
        memsafe={"requires": ["(str == NULL && size == 0) || (str != NULL && writable(str, size))",
                              "format != NULL && format is NUL-terminated",
                              "ap matches the conversions in format"],
                 "ensures":  ["size > 0 && str != NULL => initialized(str, min(size, result+1)) "
                              "&& str is NUL-terminated within size bytes"]},
    )

    # ── scanf family ──
    # glibc emits __isoc99_<name> aliases when compiled with -D_GNU_SOURCE; both
    # the plain and __isoc99_* names need contracts so callers see them.
    for name in ("sscanf", "__isoc99_sscanf"):
        out[name] = _summary(
            name,
            memsafe={"requires": ["str != NULL && str is NUL-terminated",
                                  "format != NULL && format is NUL-terminated",
                                  "each varargs pointer is non-NULL and writable for "
                                  "its corresponding conversion"],
                     "ensures":  ["assigned varargs targets are initialized for "
                                  "successful conversions (count = result, "
                                  "or result==EOF on input failure)"]},
        )
    for name in ("fscanf", "__isoc99_fscanf"):
        out[name] = _summary(
            name,
            memsafe={"requires": ["stream != NULL && stream is a valid open FILE*",
                                  "format != NULL && format is NUL-terminated",
                                  "each varargs pointer is non-NULL and writable for "
                                  "its corresponding conversion"],
                     "ensures":  ["assigned varargs targets are initialized for "
                                  "successful conversions (count = result, "
                                  "or result==EOF on input failure)"]},
        )
    for name in ("scanf", "__isoc99_scanf"):
        out[name] = _summary(
            name,
            memsafe={"requires": ["format != NULL && format is NUL-terminated",
                                  "each varargs pointer is non-NULL and writable for "
                                  "its corresponding conversion"],
                     "ensures":  ["assigned varargs targets are initialized for "
                                  "successful conversions (count = result, "
                                  "or result==EOF on input failure)"]},
        )
    for name in ("vsscanf", "__isoc99_vsscanf"):
        out[name] = _summary(
            name,
            memsafe={"requires": ["str != NULL && str is NUL-terminated",
                                  "format != NULL && format is NUL-terminated",
                                  "ap matches the conversions in format"]},
        )
    for name in ("vfscanf", "__isoc99_vfscanf"):
        out[name] = _summary(
            name,
            memsafe={"requires": ["stream != NULL && stream is a valid open FILE*",
                                  "format != NULL && format is NUL-terminated",
                                  "ap matches the conversions in format"]},
        )
    for name in ("vscanf", "__isoc99_vscanf"):
        out[name] = _summary(
            name,
            memsafe={"requires": ["format != NULL && format is NUL-terminated",
                                  "ap matches the conversions in format"]},
        )

    # ── POSIX file I/O ──
    out["open"] = _summary(
        "open",
        memsafe={"requires": ["pathname is non-NULL and points to a NUL-terminated string"],
                 "ensures":  ["result is -1 on failure or a non-negative file descriptor"]},
        memleak={"ensures": ["on success: acquires a file descriptor; "
                             "caller must close result"]},
    )
    out["close"] = _summary(
        "close",
        memsafe={"requires": ["fd is -1 or an open file descriptor not already closed"]},
        memleak={"ensures": ["releases fd if it referred to an open descriptor"]},
    )
    out["read"] = _summary(
        "read",
        memsafe={"requires": ["fd is an open file descriptor opened for reading",
                              "buf is writable for count bytes"]},
    )
    out["write"] = _summary(
        "write",
        memsafe={"requires": ["fd is an open file descriptor opened for writing",
                              "buf is readable for count bytes"]},
    )
    out["lseek64"] = _summary(
        "lseek64",
        memsafe={"requires": ["fd is an open file descriptor that supports seeking"]},
    )
    out["stat"] = _summary(
        "stat",
        memsafe={"requires": ["pathname is non-NULL and points to a NUL-terminated string",
                              "statbuf is non-NULL and writable for sizeof(struct stat) bytes"],
                 "ensures":  ["on success: initialized(statbuf, sizeof(struct stat))"]},
    )
    out["fstat"] = _summary(
        "fstat",
        memsafe={"requires": ["fd is an open file descriptor",
                              "statbuf is non-NULL and writable for sizeof(struct stat) bytes"],
                 "ensures":  ["on success: initialized(statbuf, sizeof(struct stat))"]},
    )
    out["lstat"] = _summary(
        "lstat",
        memsafe={"requires": ["pathname is non-NULL and points to a NUL-terminated string",
                              "statbuf is non-NULL and writable for sizeof(struct stat) bytes"],
                 "ensures":  ["on success: initialized(statbuf, sizeof(struct stat))"]},
    )
    out["fcntl"] = _summary(
        "fcntl",
        memsafe={"requires": ["fd is an open file descriptor",
                              "varargs match the cmd's expected argument type, if any"]},
    )
    out["memchr"] = _summary(
        "memchr",
        memsafe={"requires": ["s is readable for n bytes"],
                 "ensures":  ["result is NULL or points within s[0..n-1]"]},
    )

    # ── mmap / munmap ──
    out["mmap"] = _summary(
        "mmap",
        memsafe={"requires": ["addr == NULL || addr is page-aligned",
                              "length > 0"],
                 "ensures":  ["result == MAP_FAILED || writable(result, length)"]},
        memleak={"ensures": ["on success: acquires mapping; caller must munmap(result, length)"]},
    )
    out["munmap"] = _summary(
        "munmap",
        memsafe={"requires": ["addr != NULL && addr is page-aligned",
                              "length > 0"]},
        memleak={"ensures": ["releases mapping at [addr, addr+length)"]},
    )

    # ── directory ──
    out["opendir"] = _summary(
        "opendir",
        memsafe={"requires": ["name != NULL && name is NUL-terminated"],
                 "ensures":  ["result == NULL || result is a valid DIR*"]},
        memleak={"ensures": ["acquires DIR resource: caller must closedir(result) if non-NULL"]},
    )
    out["closedir"] = _summary(
        "closedir",
        memsafe={"requires": ["dirp != NULL && dirp is a valid open DIR*"]},
        memleak={"ensures": ["releases DIR resource"]},
    )

    # ── err family (noreturn) ──
    for n in ("err", "errx"):
        out[n] = _summary(
            n, noreturn=True,
            memsafe={"requires": ["fmt != NULL && fmt is NUL-terminated"]},
        )
    for n in ("verr", "verrx"):
        out[n] = _summary(
            n, noreturn=True,
            memsafe={"requires": ["fmt != NULL && fmt is NUL-terminated",
                                  "ap matches the conversions in fmt"]},
        )

    # ── musl internal allocator aliases ──
    out["__libc_free"] = _summary(
        "__libc_free",
        memsafe={"requires": ["p is NULL or a previously allocated pointer not yet freed"]},
        memleak={"ensures": ["releases the allocation pointed to by p if non-NULL"]},
    )
    out["__libc_realloc"] = _summary(
        "__libc_realloc",
        memsafe={
            "requires": ["ptr is NULL or previously allocated and not yet freed"],
            "ensures":  ["result is NULL or allocated for size bytes",
                         "ptr ownership transferred to result on success"],
        },
        memleak={"ensures": ["on success: acquires heap allocation tied to result; releases ptr"]},
    )

    # ── compiler-rt complex-arithmetic builtins ──
    # Emitted by clang for complex multiplication; pure arithmetic, no memory ops.
    for name in ("__mulsc3", "__muldc3", "__mulxc3"):
        out[name] = _summary(name, memsafe={})

    # ── glibc C++ runtime ──
    out["__cxa_thread_atexit_impl"] = _summary(
        "__cxa_thread_atexit_impl",
        memsafe={"requires": ["func is a valid function pointer",
                              "obj is NULL or a valid pointer"]},
    )

    # ── x86 SIMD intrinsics ──
    # These are compiler builtins that clang may leave as function calls in
    # unoptimised bitcode.  Contracts are grouped by memory access pattern.
    #
    # Loads: read from a pointer operand.
    _simd_loads_aligned = [
        # 128-bit aligned loads (require 16-byte alignment)
        "_mm_load_si128",
        # 256-bit aligned loads (require 32-byte alignment)
        "_mm256_load_si256",
        # 256-bit aligned stores (require 32-byte alignment) – listed here
        # because the contract shape is the same as aligned loads.
    ]
    _simd_loads_unaligned = [
        # 128-bit unaligned loads
        "_mm_loadu_si128",
        "_mm_loadl_epi64",     # loads low 64 bits
        # 256-bit unaligned loads
        "_mm256_loadu_si256",
    ]
    _simd_loads_masked = [
        "_mm_maskload_epi32",      # reads up to 4 ints via mask
        "_mm256_maskload_epi32",   # reads up to 8 ints via mask
    ]
    # Stores: write through a pointer operand.
    _simd_stores_aligned = [
        "_mm_store_si128",         # 16-byte aligned
        "_mm256_store_si256",      # 32-byte aligned
    ]
    _simd_stores_unaligned = [
        "_mm_storeu_si128",
        "_mm_storel_epi64",        # stores low 64 bits
        "_mm256_storeu_si256",
    ]
    # Pure register-only operations (no memory access).
    _simd_pure: list[str] = [
        # 128-bit arithmetic
        "_mm_add_epi8", "_mm_add_epi16", "_mm_add_epi32", "_mm_add_epi64",
        "_mm_sub_epi8", "_mm_sub_epi16", "_mm_sub_epi32", "_mm_sub_epi64",
        "_mm_mul_epu32",
        # 128-bit bitwise
        "_mm_and_si128", "_mm_andnot_si128", "_mm_or_si128", "_mm_xor_si128",
        # 128-bit compare
        "_mm_cmpeq_epi8", "_mm_cmpeq_epi16", "_mm_cmpeq_epi32",
        "_mm_cmpeq_epi64",
        "_mm_cmpgt_epi8", "_mm_cmpgt_epi16", "_mm_cmpgt_epi32",
        "_mm_cmplt_epi16", "_mm_cmplt_epi32",
        # 128-bit shift
        "_mm_sll_epi32", "_mm_srl_epi32",
        "_mm_slli_epi16", "_mm_slli_epi32", "_mm_slli_epi64",
        "_mm_srli_epi16", "_mm_srli_epi32", "_mm_srli_epi64",
        "_mm_srai_epi32",
        # 128-bit pack / shuffle / blend
        "_mm_packs_epi32", "_mm_packus_epi16",
        "_mm_shuffle_epi8", "_mm_blendv_epi8",
        "_mm_unpackhi_epi8", "_mm_unpackhi_epi16",
        "_mm_unpacklo_epi8", "_mm_unpacklo_epi16",
        # 128-bit set / convert / extract
        "_mm_set_epi8", "_mm_set_epi16", "_mm_set_epi32", "_mm_set_epi64x",
        "_mm_set1_epi8", "_mm_set1_epi16", "_mm_set1_epi32",
        "_mm_setzero_si128",
        "_mm_cvtsi32_si128", "_mm_cvtsi64_si128", "_mm_cvtsi128_si32",
        "_mm_movemask_epi8",
        "_mm_avg_epu8", "_mm_min_epi16",
        # 256-bit arithmetic
        "_mm256_add_epi8", "_mm256_add_epi16", "_mm256_add_epi32",
        "_mm256_add_epi64",
        "_mm256_sub_epi8", "_mm256_sub_epi16", "_mm256_sub_epi32",
        "_mm256_sub_epi64",
        "_mm256_mul_epu32",
        # 256-bit bitwise
        "_mm256_and_si256", "_mm256_andnot_si256",
        "_mm256_or_si256", "_mm256_xor_si256",
        # 256-bit compare
        "_mm256_cmpeq_epi8", "_mm256_cmpeq_epi16", "_mm256_cmpeq_epi32",
        "_mm256_cmpeq_epi64",
        # 256-bit shift
        "_mm256_slli_epi16", "_mm256_slli_epi32", "_mm256_slli_epi64",
        "_mm256_srli_epi16", "_mm256_srli_epi32", "_mm256_srli_epi64",
        "_mm256_sllv_epi32", "_mm256_srlv_epi32",
        # 256-bit pack / shuffle / broadcast / cast
        "_mm256_shuffle_epi8",
        "_mm256_broadcastb_epi8", "_mm256_broadcastw_epi16",
        "_mm256_broadcastd_epi32", "_mm256_broadcastq_epi64",
        "_mm256_castsi128_si256", "_mm256_castsi256_si128",
        "_mm256_unpacklo_epi64",
        "_mm256_permutevar8x32_epi32",
        # 256-bit set / convert
        "_mm256_set_epi64x", "_mm256_setzero_si256",
        "_mm256_cvtepi8_epi32",
    ]

    for name in _simd_loads_aligned:
        out[name] = _summary(
            name,
            memsafe={"requires": ["src is non-NULL",
                                  "src is aligned to vector width (16 or 32 bytes)",
                                  "src is readable for vector width bytes"]},
        )
    for name in _simd_loads_unaligned:
        out[name] = _summary(
            name,
            memsafe={"requires": ["src is non-NULL",
                                  "src is readable for vector width bytes"]},
        )
    for name in _simd_loads_masked:
        out[name] = _summary(
            name,
            memsafe={"requires": ["src is non-NULL",
                                  "src is readable for vector width bytes"]},
        )
    for name in _simd_stores_aligned:
        out[name] = _summary(
            name,
            memsafe={"requires": ["dst is non-NULL",
                                  "dst is aligned to vector width (16 or 32 bytes)",
                                  "dst is writable for vector width bytes"],
                     "modifies": ["dst[0..vector_width-1]"]},
        )
    for name in _simd_stores_unaligned:
        out[name] = _summary(
            name,
            memsafe={"requires": ["dst is non-NULL",
                                  "dst is writable for vector width bytes"],
                     "modifies": ["dst[0..vector_width-1]"]},
        )
    for name in _simd_pure:
        out[name] = _summary(name, memsafe={})

    # Register mangled variants that appear in consumer bitcode.
    # The contract is identical — only the abilist key differs.
    _simd_mangled_map: dict[str, list[str]] = {
        # Loads
        "_mm_load_si128": ["_ZL15_mm_load_si128PKDv2_x"],
        "_mm_loadu_si128": ["_ZL15_mm_loadu_si128PKDv2_x"],
        "_mm_loadl_epi64": ["_ZL15_mm_loadl_epi64PKDv2_x"],
        "_mm256_load_si256": ["_ZL18_mm256_load_si256PKDv4_x"],
        "_mm256_loadu_si256": ["_ZL19_mm256_loadu_si256PKDv4_x"],
        "_mm_maskload_epi32": ["_ZL19_mm_maskload_epi32PKiDv2_x"],
        "_mm256_maskload_epi32": ["_ZL22_mm256_maskload_epi32PKiDv4_x"],
        # Stores
        "_mm_store_si128": ["_ZL16_mm_store_si128PDv2_xS_"],
        "_mm_storeu_si128": ["_ZL16_mm_storeu_si128PDv2_xS_"],
        "_mm_storel_epi64": ["_ZL16_mm_storel_epi64PDv2_xS_"],
        "_mm256_store_si256": ["_ZL19_mm256_store_si256PDv4_xS_"],
        "_mm256_storeu_si256": ["_ZL20_mm256_storeu_si256PDv4_xS_"],
        # Pure
        "_mm_add_epi8": ["_ZL12_mm_add_epi8Dv2_xS_"],
        "_mm_add_epi16": ["_ZL13_mm_add_epi16Dv2_xS_"],
        "_mm_add_epi32": ["_ZL13_mm_add_epi32Dv2_xS_"],
        "_mm_add_epi64": ["_ZL13_mm_add_epi64Dv2_xS_"],
        "_mm_sub_epi8": ["_ZL12_mm_sub_epi8Dv2_xS_"],
        "_mm_sub_epi16": ["_ZL13_mm_sub_epi16Dv2_xS_"],
        "_mm_sub_epi32": ["_ZL13_mm_sub_epi32Dv2_xS_"],
        "_mm_sub_epi64": ["_ZL13_mm_sub_epi64Dv2_xS_"],
        "_mm_mul_epu32": ["_ZL13_mm_mul_epu32Dv2_xS_"],
        "_mm_and_si128": ["_ZL13_mm_and_si128Dv2_xS_"],
        "_mm_andnot_si128": ["_ZL16_mm_andnot_si128Dv2_xS_"],
        "_mm_or_si128": ["_ZL12_mm_or_si128Dv2_xS_"],
        "_mm_xor_si128": ["_ZL13_mm_xor_si128Dv2_xS_"],
        "_mm_cmpeq_epi8": ["_ZL14_mm_cmpeq_epi8Dv2_xS_"],
        "_mm_cmpeq_epi16": ["_ZL15_mm_cmpeq_epi16Dv2_xS_"],
        "_mm_cmpeq_epi32": ["_ZL15_mm_cmpeq_epi32Dv2_xS_"],
        "_mm_cmpeq_epi64": ["_ZL15_mm_cmpeq_epi64Dv2_xS_"],
        "_mm_cmpgt_epi8": ["_ZL14_mm_cmpgt_epi8Dv2_xS_"],
        "_mm_cmpgt_epi16": ["_ZL15_mm_cmpgt_epi16Dv2_xS_"],
        "_mm_cmpgt_epi32": ["_ZL15_mm_cmpgt_epi32Dv2_xS_"],
        "_mm_cmplt_epi16": ["_ZL15_mm_cmplt_epi16Dv2_xS_"],
        "_mm_cmplt_epi32": ["_ZL15_mm_cmplt_epi32Dv2_xS_"],
        "_mm_sll_epi32": ["_ZL13_mm_sll_epi32Dv2_xS_"],
        "_mm_srl_epi32": ["_ZL13_mm_srl_epi32Dv2_xS_"],
        "_mm_slli_epi16": ["_ZL14_mm_slli_epi16Dv2_xi"],
        "_mm_slli_epi32": ["_ZL14_mm_slli_epi32Dv2_xi"],
        "_mm_slli_epi64": ["_ZL14_mm_slli_epi64Dv2_xi"],
        "_mm_srli_epi16": ["_ZL14_mm_srli_epi16Dv2_xi"],
        "_mm_srli_epi32": ["_ZL14_mm_srli_epi32Dv2_xi"],
        "_mm_srli_epi64": ["_ZL14_mm_srli_epi64Dv2_xi"],
        "_mm_srai_epi32": ["_ZL14_mm_srai_epi32Dv2_xi"],
        "_mm_packs_epi32": ["_ZL15_mm_packs_epi32Dv2_xS_"],
        "_mm_packus_epi16": ["_ZL16_mm_packus_epi16Dv2_xS_"],
        "_mm_shuffle_epi8": ["_ZL16_mm_shuffle_epi8Dv2_xS_"],
        "_mm_blendv_epi8": ["_ZL15_mm_blendv_epi8Dv2_xS_S_"],
        "_mm_unpackhi_epi8": ["_ZL17_mm_unpackhi_epi8Dv2_xS_"],
        "_mm_unpackhi_epi16": ["_ZL18_mm_unpackhi_epi16Dv2_xS_"],
        "_mm_unpacklo_epi8": ["_ZL17_mm_unpacklo_epi8Dv2_xS_"],
        "_mm_unpacklo_epi16": ["_ZL18_mm_unpacklo_epi16Dv2_xS_"],
        "_mm_set_epi8": ["_ZL12_mm_set_epi8cccccccccccccccc"],
        "_mm_set_epi16": ["_ZL13_mm_set_epi16ssssssss"],
        "_mm_set_epi32": ["_ZL13_mm_set_epi32iiii"],
        "_mm_set_epi64x": ["_ZL14_mm_set_epi64xxx"],
        "_mm_set1_epi8": ["_ZL13_mm_set1_epi8c"],
        "_mm_set1_epi16": ["_ZL14_mm_set1_epi16s"],
        "_mm_set1_epi32": ["_ZL14_mm_set1_epi32i"],
        "_mm_setzero_si128": ["_ZL17_mm_setzero_si128v"],
        "_mm_cvtsi32_si128": ["_ZL17_mm_cvtsi32_si128i"],
        "_mm_cvtsi64_si128": ["_ZL17_mm_cvtsi64_si128x"],
        "_mm_cvtsi128_si32": ["_ZL17_mm_cvtsi128_si32Dv2_x"],
        "_mm_movemask_epi8": ["_ZL17_mm_movemask_epi8Dv2_x"],
        "_mm_avg_epu8": ["_ZL12_mm_avg_epu8Dv2_xS_"],
        "_mm_min_epi16": ["_ZL13_mm_min_epi16Dv2_xS_"],
        # 256-bit pure
        "_mm256_add_epi8": ["_ZL15_mm256_add_epi8Dv4_xS_"],
        "_mm256_add_epi16": ["_ZL16_mm256_add_epi16Dv4_xS_"],
        "_mm256_add_epi32": ["_ZL16_mm256_add_epi32Dv4_xS_"],
        "_mm256_add_epi64": ["_ZL16_mm256_add_epi64Dv4_xS_"],
        "_mm256_sub_epi8": ["_ZL15_mm256_sub_epi8Dv4_xS_"],
        "_mm256_sub_epi16": ["_ZL16_mm256_sub_epi16Dv4_xS_"],
        "_mm256_sub_epi32": ["_ZL16_mm256_sub_epi32Dv4_xS_"],
        "_mm256_sub_epi64": ["_ZL16_mm256_sub_epi64Dv4_xS_"],
        "_mm256_mul_epu32": ["_ZL16_mm256_mul_epu32Dv4_xS_"],
        "_mm256_and_si256": ["_ZL16_mm256_and_si256Dv4_xS_"],
        "_mm256_andnot_si256": ["_ZL19_mm256_andnot_si256Dv4_xS_"],
        "_mm256_or_si256": ["_ZL15_mm256_or_si256Dv4_xS_"],
        "_mm256_xor_si256": ["_ZL16_mm256_xor_si256Dv4_xS_"],
        "_mm256_cmpeq_epi8": ["_ZL17_mm256_cmpeq_epi8Dv4_xS_"],
        "_mm256_cmpeq_epi16": ["_ZL18_mm256_cmpeq_epi16Dv4_xS_"],
        "_mm256_cmpeq_epi32": ["_ZL18_mm256_cmpeq_epi32Dv4_xS_"],
        "_mm256_cmpeq_epi64": ["_ZL18_mm256_cmpeq_epi64Dv4_xS_"],
        "_mm256_slli_epi16": ["_ZL17_mm256_slli_epi16Dv4_xi"],
        "_mm256_slli_epi32": ["_ZL17_mm256_slli_epi32Dv4_xi"],
        "_mm256_slli_epi64": ["_ZL17_mm256_slli_epi64Dv4_xi"],
        "_mm256_srli_epi16": ["_ZL17_mm256_srli_epi16Dv4_xi"],
        "_mm256_srli_epi32": ["_ZL17_mm256_srli_epi32Dv4_xi"],
        "_mm256_srli_epi64": ["_ZL17_mm256_srli_epi64Dv4_xi"],
        "_mm256_sllv_epi32": ["_ZL17_mm256_sllv_epi32Dv4_xS_"],
        "_mm256_srlv_epi32": ["_ZL17_mm256_srlv_epi32Dv4_xS_"],
        "_mm256_shuffle_epi8": ["_ZL19_mm256_shuffle_epi8Dv4_xS_"],
        "_mm256_broadcastb_epi8": ["_ZL22_mm256_broadcastb_epi8Dv2_x"],
        "_mm256_broadcastw_epi16": ["_ZL23_mm256_broadcastw_epi16Dv2_x"],
        "_mm256_broadcastd_epi32": ["_ZL23_mm256_broadcastd_epi32Dv2_x"],
        "_mm256_broadcastq_epi64": ["_ZL23_mm256_broadcastq_epi64Dv2_x"],
        "_mm256_castsi128_si256": ["_ZL22_mm256_castsi128_si256Dv2_x"],
        "_mm256_castsi256_si128": ["_ZL22_mm256_castsi256_si128Dv4_x"],
        "_mm256_unpacklo_epi64": ["_ZL21_mm256_unpacklo_epi64Dv4_xS_"],
        "_mm256_permutevar8x32_epi32": [
            "_ZL27_mm256_permutevar8x32_epi32Dv4_xS_",
        ],
        "_mm256_set_epi64x": ["_ZL17_mm256_set_epi64xxxxx"],
        "_mm256_setzero_si256": ["_ZL20_mm256_setzero_si256v"],
        "_mm256_cvtepi8_epi32": ["_ZL20_mm256_cvtepi8_epi32Dv2_x"],
    }

    for base_name, mangled_names in _simd_mangled_map.items():
        base = out.get(base_name)
        if base is None:
            continue
        for mname in mangled_names:
            alias = CodeContractSummary(
                function=mname, properties=list(base.properties),
            )
            for prop in base.properties:
                alias.requires[prop] = list(base.requires.get(prop, []))
                alias.ensures[prop] = list(base.ensures.get(prop, []))
                alias.modifies[prop] = list(base.modifies.get(prop, []))
            out[mname] = alias

    return out


STDLIB_CONTRACTS: dict[str, CodeContractSummary] = _build_libc_contracts()
