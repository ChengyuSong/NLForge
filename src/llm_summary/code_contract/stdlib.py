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

    return out


STDLIB_CONTRACTS: dict[str, CodeContractSummary] = _build_libc_contracts()
