"""Shared git-based file-exploration tools for all LLM agents.

Uses git plumbing (ls-tree, show, grep) so results are scoped to tracked
objects — no symlink-escape or path-traversal concerns.  All subprocess
calls use list argv (never shell=True) and ``--`` separators to prevent
argument injection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _safe_ref(ref: str) -> str:
    """Reject refs that could be interpreted as flags."""
    if ref.startswith("-"):
        raise ValueError(f"Invalid ref: {ref}")
    return ref


def _safe_path(path: str) -> str:
    """Reject paths that could be interpreted as flags."""
    if path.startswith("-"):
        raise ValueError(f"Invalid path: {path}")
    return path


def _run_git(
    repo: Path, args: list[str], *, timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run a git command in *repo*, returning CompletedProcess."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=timeout,
    )


class GitTools:
    """Read, list, and grep tracked files via git plumbing."""

    def __init__(self, repo_path: Path, ref: str = "HEAD") -> None:
        self.repo = Path(repo_path).resolve()
        self.default_ref = _safe_ref(ref)

    # ------------------------------------------------------------------
    # git_show  (git show <ref>:<path>)
    # ------------------------------------------------------------------

    def git_show(
        self,
        file_path: str,
        max_lines: int = 200,
        start_line: int = 1,
        ref: str | None = None,
    ) -> dict[str, Any]:
        """Read a tracked file by its blob object."""
        ref = _safe_ref(ref or self.default_ref)
        file_path = _safe_path(file_path)

        if start_line < 1:
            return {"error": f"start_line must be >= 1, got {start_line}"}

        proc = _run_git(self.repo, ["show", f"{ref}:{file_path}"])
        if proc.returncode != 0:
            msg = proc.stderr.strip()
            if "does not exist" in msg or "not exist" in msg:
                return {"error": f"File not found: {file_path} (ref={ref})"}
            return {"error": msg or f"git show failed (rc={proc.returncode})"}

        all_lines = proc.stdout.splitlines()
        total = len(all_lines)

        if start_line > total:
            return {
                "error": (
                    f"start_line {start_line} exceeds file length"
                    f" ({total} lines)"
                ),
            }

        selected = all_lines[start_line - 1 : start_line - 1 + max_lines]
        numbered = [
            f"{start_line + i:4d}: {line}"
            for i, line in enumerate(selected)
        ]

        truncated = (start_line - 1 + max_lines) < total
        if truncated:
            numbered.append(
                f"\n... (truncated after {max_lines} lines,"
                f" use start_line={start_line + len(selected)} to continue)"
            )

        return {
            "content": "\n".join(numbered),
            "path": file_path,
            "ref": ref,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "lines_read": len(selected),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # git_ls_tree  (git ls-tree <ref> <path>)
    # ------------------------------------------------------------------

    def git_ls_tree(
        self,
        dir_path: str = ".",
        ref: str | None = None,
    ) -> dict[str, Any]:
        """List tracked entries in a directory via git ls-tree."""
        ref = _safe_ref(ref or self.default_ref)
        dir_path = _safe_path(dir_path)

        tree_path = dir_path.rstrip("/")
        if tree_path in (".", ""):
            args = ["ls-tree", "-l", ref]
        else:
            args = ["ls-tree", "-l", ref, "--", f"{tree_path}/"]

        proc = _run_git(self.repo, args)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or "git ls-tree failed"}

        prefix = f"{tree_path}/" if tree_path not in (".", "") else ""
        directories: list[str] = []
        files: list[str] = []

        for line in (proc.stdout.strip().splitlines()
                     if proc.stdout.strip() else [])[:500]:
            # format: <mode> <type> <hash> <size>\t<path>
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            meta, path = parts
            name = path[len(prefix):] if path.startswith(prefix) else path
            meta_parts = meta.split()
            obj_type = meta_parts[1] if len(meta_parts) >= 2 else "blob"
            if obj_type == "tree":
                directories.append(f"{name}/")
            else:
                size = meta_parts[3] if len(meta_parts) >= 4 else "-"
                files.append(f"{name}  ({size} bytes)")

        return {
            "path": dir_path,
            "ref": ref,
            "directories": sorted(directories),
            "files": sorted(files),
            "total": len(directories) + len(files),
        }

    # ------------------------------------------------------------------
    # git_grep  (git grep)
    # ------------------------------------------------------------------

    def git_grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        glob: str | None = None,
        max_results: int = 50,
        context: int = 0,
        case_insensitive: bool = False,
        ref: str | None = None,
    ) -> dict[str, Any]:
        """Search tracked file contents using git grep."""
        ref = _safe_ref(ref or self.default_ref)
        path = _safe_path(path)

        cmd: list[str] = [
            "grep", "-P", "-n", f"--max-count={max_results}",
        ]
        if context > 0:
            cmd.append(f"-C{context}")
        if case_insensitive:
            cmd.append("-i")
        cmd.append("-e")
        cmd.append(pattern)
        cmd.append(ref)

        # Path specs after --
        cmd.append("--")
        if glob:
            pathspecs_dir = path if path not in (".", "") else ""
            if pathspecs_dir:
                cmd.append(f"{pathspecs_dir}/{glob}")
            else:
                cmd.append(glob)
        elif path not in (".", ""):
            cmd.append(path)

        proc = _run_git(self.repo, cmd, timeout=30)

        # git grep returns 1 for no matches (not an error)
        if proc.returncode not in (0, 1):
            return {"error": proc.stderr.strip() or "git grep failed"}

        lines = proc.stdout.strip().splitlines() if proc.stdout else []

        # Strip "ref:" prefix (e.g. "HEAD:src/foo.c:42:...")
        ref_prefix = f"{ref}:"
        cleaned: list[str] = []
        for line in lines[:max_results]:
            if line.startswith(ref_prefix):
                line = line[len(ref_prefix):]
            cleaned.append(line)

        return {
            "pattern": pattern,
            "path": path,
            "ref": ref,
            "match_count": len(cleaned),
            "matches": cleaned,
            "truncated": len(lines) > max_results,
        }

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name."""
        handlers = {
            "git_show": self._handle_git_show,
            "git_ls_tree": self._handle_git_ls_tree,
            "git_grep": self._handle_git_grep,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown git tool: {tool_name}"}
        return handler(tool_input)

    def _handle_git_show(self, inp: dict[str, Any]) -> dict[str, Any]:
        return self.git_show(
            file_path=inp["file_path"],
            max_lines=inp.get("max_lines", 200),
            start_line=inp.get("start_line", 1),
        )

    def _handle_git_ls_tree(self, inp: dict[str, Any]) -> dict[str, Any]:
        return self.git_ls_tree(
            dir_path=inp.get("dir_path", "."),
        )

    def _handle_git_grep(self, inp: dict[str, Any]) -> dict[str, Any]:
        return self.git_grep(
            pattern=inp["pattern"],
            path=inp.get("path", "."),
            glob=inp.get("glob"),
            max_results=inp.get("max_results", 50),
            context=inp.get("context", 0),
            case_insensitive=inp.get("case_insensitive", False),
        )


# ======================================================================
# Tool definitions (Anthropic tool-use schema)
# ======================================================================

GIT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "git_show",
        "description": (
            "Read a tracked file from the project git repository "
            "(like git show HEAD:<path>). "
            "Paths are relative to the repo root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path (e.g. 'src/main.c').",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Max lines to read (default 200).",
                    "default": 200,
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed line to start from.",
                    "default": 1,
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "git_ls_tree",
        "description": (
            "List tracked files and directories in the repository "
            "(like git ls-tree). Shows only committed/staged content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "Relative path (e.g. '.', 'src').",
                    "default": ".",
                },
            },
        },
    },
    {
        "name": "git_grep",
        "description": (
            "Search tracked file contents using git grep. "
            "Use to find function definitions, macro usages, "
            "documentation, or any text pattern in the project."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory or file to search in.",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. '*.c', '*.h').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches (default 50).",
                    "default": 50,
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context around each match.",
                    "default": 0,
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search.",
                    "default": False,
                },
            },
            "required": ["pattern"],
        },
    },
]
