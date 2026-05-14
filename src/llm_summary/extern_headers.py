"""Extract external function declaration headers via clang -E preprocessing.

For each source file in compile_commands.json, runs the preprocessor and parses
line markers (``# <line> "<file>"``) to determine which header file declares
each external function.  This lets us map e.g. ``crc32`` -> ``/usr/include/zlib.h``.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

# Regex for preprocessor line markers:  # 123 "/usr/include/zlib.h" 3 4
_LINE_MARKER_RE = re.compile(r'^#\s+\d+\s+"([^"]+)"')

# Regex for extern function declarations.  Uses \bextern (not ^extern) so it
# matches multiple declarations on a single preprocessor output line — common
# when headers expand macros (e.g. pcre2.h emits all decls on one line).
_EXTERN_FUNC_RE = re.compile(
    r"\bextern\s+"            # 'extern' at a word boundary
    r"[\w\s\*]+"              # return type (words, spaces, pointers)
    r"\b(\w+)\s*\("           # function name followed by '('
)

# Flags to strip when converting a compile command to preprocessor-only
_STRIP_FLAGS = {"-c", "-save-temps", "-save-temps=obj", "-save-temps=cwd"}
_STRIP_PREFIX = ("-flto",)

# ---------------------------------------------------------------------------
# Well-known standard library headers (libc, libm, POSIX, pthreads)
# Maps header basename -> library name.  Used by import-dep to skip
# functions that belong to the standard library (handled by init-stdlib).
# ---------------------------------------------------------------------------
_STDLIB_HEADERS: dict[str, str] = {}

# C standard library (libc)
for _h in (
    "assert.h", "complex.h", "ctype.h", "errno.h", "fenv.h", "float.h",
    "inttypes.h", "iso646.h", "limits.h", "locale.h", "setjmp.h",
    "signal.h", "stdalign.h", "stdarg.h", "stdatomic.h", "stdbool.h",
    "stddef.h", "stdint.h", "stdio.h", "stdlib.h", "stdnoreturn.h",
    "string.h", "strings.h", "tgmath.h", "time.h", "uchar.h", "wchar.h",
    "wctype.h",
):
    _STDLIB_HEADERS[_h] = "libc"

# POSIX / system headers (still libc on Linux)
for _h in (
    "aio.h", "arpa/inet.h", "cpio.h", "dirent.h", "dlfcn.h", "fcntl.h",
    "fmtmsg.h", "fnmatch.h", "ftw.h", "glob.h", "grp.h", "iconv.h",
    "ifaddrs.h", "langinfo.h", "libgen.h", "link.h", "mqueue.h",
    "net/if.h", "netdb.h", "netinet/in.h", "netinet/tcp.h", "nl_types.h",
    "poll.h", "pthread.h", "pwd.h", "regex.h", "sched.h", "search.h",
    "semaphore.h", "spawn.h", "syslog.h", "tar.h", "termios.h",
    "unistd.h", "utime.h", "utmpx.h", "wordexp.h",
):
    _STDLIB_HEADERS[_h] = "libc"

# sys/ headers
for _h in (
    "sys/ipc.h", "sys/mman.h", "sys/msg.h", "sys/resource.h", "sys/select.h",
    "sys/sem.h", "sys/shm.h", "sys/socket.h", "sys/stat.h", "sys/statvfs.h",
    "sys/time.h", "sys/times.h", "sys/types.h", "sys/uio.h", "sys/un.h",
    "sys/utsname.h", "sys/wait.h", "sys/epoll.h", "sys/eventfd.h",
    "sys/ioctl.h", "sys/sendfile.h", "sys/signalfd.h", "sys/timerfd.h",
    "sys/prctl.h", "sys/ptrace.h", "sys/xattr.h", "sys/file.h",
    "sys/mount.h", "sys/swap.h", "sys/reboot.h", "sys/sysinfo.h",
):
    _STDLIB_HEADERS[_h] = "libc"

# libm (math library — on Linux, part of glibc but linked separately)
for _h in ("math.h",):
    _STDLIB_HEADERS[_h] = "libm"

# Linux-specific headers (still glibc/kernel)
for _h in (
    "err.h", "error.h", "endian.h", "getopt.h", "alloca.h", "mntent.h",
    "paths.h", "resolv.h", "sysexits.h", "features.h",
    "linux/capability.h", "linux/if.h", "linux/limits.h",
):
    _STDLIB_HEADERS[_h] = "libc"

del _h


def is_stdlib_header(header_path: str) -> bool:
    """Check if a header path belongs to a well-known standard library."""
    return classify_header(header_path) is not None


def classify_header(header_path: str) -> str | None:
    """Classify a header path as a known library or None.

    Returns "libc", "libm", or None for third-party/unknown headers.
    Matches by basename or common subpath (e.g. "sys/mman.h").
    """
    p = header_path
    # Strip /usr/include/ or /usr/include/<arch>/ prefix
    for prefix in ("/usr/include/", "/usr/local/include/"):
        if p.startswith(prefix):
            rel = p[len(prefix):]
            # Strip arch-specific dir like x86_64-linux-gnu/
            parts = rel.split("/")
            if parts and "-linux-" in parts[0]:
                rel = "/".join(parts[1:])
            # Also handle bits/ and gnu/ subdirs (internal glibc headers)
            if rel.startswith(("bits/", "gnu/", "asm/", "asm-generic/", "linux/")):
                return "libc"
            if rel in _STDLIB_HEADERS:
                return _STDLIB_HEADERS[rel]
            # basename fallback
            basename = Path(rel).name
            if basename in _STDLIB_HEADERS:
                return _STDLIB_HEADERS[basename]
            return None
    return None


def extract_extern_headers(
    compile_commands_path: str | Path,
    project_root: str | Path | None = None,
    source_files: list[str] | None = None,
    verbose: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Run clang -E on source files and map external function names to headers.

    Args:
        compile_commands_path: Path to compile_commands.json (host-remapped).
        project_root: Project source root (to distinguish project vs system headers).
        source_files: Optional subset of source files to process.
            If None, processes all C/C++ files from compile_commands.
        verbose: Print progress.

    Returns:
        Tuple of (header_map, failed_sources) where header_map maps function
        name -> header path for functions declared in non-project headers, and
        failed_sources is a list of source files where preprocessing failed
        (e.g. missing build directory or generated headers).
    """
    cc_path = Path(compile_commands_path)
    with open(cc_path, encoding="utf-8") as f:
        entries = json.load(f)

    project_prefixes = _build_project_prefixes(project_root)

    # Build source -> (compiler, flags, directory) mapping
    file_cmds: dict[str, tuple[str, list[str], str]] = {}
    for entry in entries:
        src = entry.get("file", "")
        if not src:
            continue
        directory = entry.get("directory", "")
        if "arguments" in entry:
            args = list(entry["arguments"])
        elif "command" in entry:
            args = shlex.split(entry["command"])
        else:
            continue

        compiler, flags = _make_preprocess_cmd(args, src)
        if compiler:
            abs_src = src if Path(src).is_absolute() else str(Path(directory) / src)
            file_cmds[abs_src] = (compiler, flags, directory)

    # Filter to requested source files
    c_exts = {".c", ".cpp", ".cc", ".cxx", ".c++"}
    if source_files is not None:
        targets = [f for f in source_files if f in file_cmds]
    else:
        targets = [f for f in file_cmds if Path(f).suffix.lower() in c_exts]

    if verbose:
        print(f"[extern-headers] Preprocessing {len(targets)} source files")

    header_map: dict[str, str] = {}
    failed_sources: list[str] = []

    for src in targets:
        compiler, flags, directory = file_cmds[src]

        pp_result = _run_preprocessor(compiler, flags, src, directory, project_prefixes, verbose)
        if pp_result is None:
            failed_sources.append(src)
        else:
            header_map.update(pp_result)

    return header_map, failed_sources


def _build_project_prefixes(project_root: str | Path | None) -> list[str]:
    """Build list of path prefixes that identify project-local headers."""
    prefixes = []
    if project_root:
        prefixes.append(str(Path(project_root).resolve()))
    return prefixes


def _is_project_header(path: str, project_prefixes: list[str]) -> bool:
    """Check if a header path belongs to the project (not a system/library header)."""
    if not project_prefixes:
        # Without project_root, treat /usr/ and /lib/ as system
        return not (path.startswith("/usr/") or path.startswith("/lib/"))
    return any(path.startswith(p) for p in project_prefixes)


def _make_preprocess_cmd(
    args: list[str], source_file: str
) -> tuple[str | None, list[str]]:
    """Convert a compile command to a preprocessor-only command.

    Returns (compiler, flags) where flags include -E and the source file.
    """
    if not args:
        return None, []

    compiler = args[0]
    flags = ["-E"]
    skip_next = False

    for i, arg in enumerate(args):
        if i == 0:
            continue
        if skip_next:
            skip_next = False
            continue

        # Strip output flag
        if arg == "-o":
            skip_next = True
            continue
        if arg.startswith("-o"):
            continue

        # Strip flags incompatible with -E
        if arg in _STRIP_FLAGS:
            continue
        if any(arg.startswith(p) for p in _STRIP_PREFIX):
            continue

        # Skip dependency file flags
        if arg in ("-MF", "-MT", "-MQ"):
            skip_next = True
            continue
        if arg.startswith("-M"):
            continue

        flags.append(arg)

    # Append at end so it overrides any earlier -Werror: flags like
    # -stdlib=libc++ are valid for compilation but unused by -E.
    flags.append("-Wno-unused-command-line-argument")

    return compiler, flags


def _run_preprocessor(
    compiler: str,
    flags: list[str],
    source_file: str,
    directory: str,
    project_prefixes: list[str],
    verbose: bool,
) -> dict[str, str] | None:
    """Run the preprocessor on a single file and parse extern declarations.

    Returns None on failure (missing build dir, non-zero exit, timeout).
    Returns an empty dict when preprocessing succeeded but found no externs.
    """
    cmd = [compiler] + flags + [source_file]

    try:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            cmd,
            cwd=directory or None,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        if verbose:
            print(f"[extern-headers]   {Path(source_file).name}: preprocessor failed: {e}")
        return None

    if result.returncode != 0:
        if verbose:
            stderr_line = result.stderr.strip().split("\n")[0] if result.stderr else ""
            print(f"[extern-headers]   {Path(source_file).name}: preprocessor error: {stderr_line}")
        return None

    return _parse_preprocessor_output(result.stdout, project_prefixes)


def _parse_preprocessor_output(
    text: str, project_prefixes: list[str]
) -> dict[str, str]:
    """Parse preprocessor output to extract extern function -> header mappings."""
    header_map: dict[str, str] = {}
    current_file = ""

    for line in text.split("\n"):
        # Track file context from line markers
        m = _LINE_MARKER_RE.match(line)
        if m:
            current_file = m.group(1)
            continue

        # Only look for extern declarations in non-project headers
        if not current_file or _is_project_header(current_file, project_prefixes):
            continue

        for m in _EXTERN_FUNC_RE.finditer(line):
            func_name = m.group(1)
            if func_name not in header_map:
                header_map[func_name] = current_file

    return header_map
