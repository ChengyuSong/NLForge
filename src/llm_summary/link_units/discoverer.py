"""ReAct agent for link-unit discovery.

Uses LLM-driven exploration to identify link units in builds that
cannot be parsed deterministically (e.g., autotools, custom Makefiles).

Safe tools (read_file, list_dir, find_files, parse_ninja_targets, run_ar_t)
run on the host with path sandboxing via BuildTools.

Arbitrary shell commands run inside a Docker container with read-only mounts.
"""

import json
import os
import subprocess
from pathlib import Path

from ..builder.llm_utils import (
    compress_stale_results,
    estimate_messages_tokens,
    track_tool_result,
    truncate_messages,
)
from ..builder.tools import BuildTools
from ..llm.base import LLMBackend
from .skills import parse_ninja_targets, prescan_build_dir, run_ar_t
from .tool_definitions import DISCOVERY_TOOL_DEFINITIONS

# ReAct loop parameters
BASE_TURNS = 12          # initial exploration overhead
PER_TARGET_TURNS = 3     # turns per unresolved target (list + read Makefile + grep + verify)
MIN_TURNS = 20
MAX_CONTEXT_TOKENS = 100000
TURNS_LOW_WARNING = 7
TURNS_EXTENSION = 10     # extra turns granted per request_more_turns call
MAX_TURN_EXTENSIONS = 10 # maximum extensions allowed

# Docker paths (same convention as builder)
DOCKER_WORKSPACE_SRC = "/workspace/src"
DOCKER_WORKSPACE_BUILD = "/workspace/build"
DOCKER_COMMAND_TIMEOUT = 120

SYSTEM_PROMPT = """\
You are a build-system analyst. Discover link units (libraries and executables) \
and identify which object files belong to each.

## Layout
- Source: `/workspace/src` (or `read_file` with relative paths)
- Build: `/workspace/build` (or `read_file` with `build/` prefix)
- `run_command` runs in Docker with read-only mounts. Use freely for grep, \
find, make -n, nm, readelf, etc.

## Approach
- Inspect Makefiles and build rules to find link targets and their objects.
- Use `run_ar_t` for archive members, `parse_ninja_targets` for Ninja builds.
- Focus on the project's own link units, not system libraries.
- You do NOT need to find .bc files — that is a post-processing step.

## Rules
- Be efficient — you have a limited turn budget.
- Call `finish()` as soon as you have identified the link units.
- Prefer `finish(status='partial')` over running out of turns.
- Call `request_more_turns` if you are still making progress but need more turns.
"""


class LinkUnitDiscoverer:
    """Discover link units using LLM-driven exploration."""

    def __init__(
        self,
        llm: LLMBackend,
        container_image: str = "llm-summary-builder:latest",
        verbose: bool = False,
        log_file: str | None = None,
    ):
        self.llm = llm
        self.container_image = container_image
        self.verbose = verbose
        self.log_file = log_file

    def discover(
        self,
        project_name: str,
        project_path: Path,
        build_dir: Path,
        build_system: str | None = None,
        heuristic_result: dict | None = None,
        unresolved_objects: list[str] | None = None,
    ) -> dict:
        """Run the ReAct discovery loop.

        Args:
            project_name: Name of the project
            project_path: Path to project source
            build_dir: Path to build directory
            build_system: Optional hint about build system type
            heuristic_result: Pre-computed link units from heuristic pass
            unresolved_objects: Executables that heuristics couldn't resolve
        """
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir).resolve()
        self.heuristic_result = heuristic_result

        file_tools = BuildTools(self.project_path, self.build_dir)

        # Dynamic turn budget based on unresolved count
        if unresolved_objects:
            n_unresolved = len(unresolved_objects)
        else:
            prescan = prescan_build_dir(self.build_dir, verbose=self.verbose)
            n_unresolved = prescan["estimated_targets"]

        self.max_turns = max(
            MIN_TURNS,
            int(BASE_TURNS + PER_TARGET_TURNS * n_unresolved),
        )

        if self.verbose:
            print(f"[link-units] Starting agent for {project_name}")
            print(f"[link-units] Build dir: {build_dir}")
            print(f"[link-units] Turn budget: {self.max_turns}")

        # Build initial context message
        context_parts = [
            f"Project: {project_name}",
            f"Source: /workspace/src",
            f"Build: /workspace/build",
        ]
        if build_system:
            context_parts.append(f"Build system: {build_system}")

        if heuristic_result and unresolved_objects:
            # Focused mode: only resolve the unresolved executables
            already = [f"  - {u['type']} {u['name']}" for u in heuristic_result.get("link_units", [])]
            initial_msg = (
                "\n".join(context_parts)
                + "\n\nAlready discovered:\n" + "\n".join(already)
                + "\n\nUnresolved ELF executables:\n"
                + "\n".join(f"  - {o}" for o in unresolved_objects)
                + "\n\nFor each, find its object files and library deps from the build system. "
                "Skip build tools or test utilities. "
                "Call finish() with ONLY newly discovered units."
            )
        else:
            initial_msg = (
                "\n".join(context_parts)
                + "\n\nDiscover all link units in this build. "
                "Start by exploring the build directory."
            )

        messages = [{"role": "user", "content": initial_msg}]

        # Initial LLM call
        response = self.llm.complete_with_tools(
            messages=messages,
            tools=DISCOVERY_TOOL_DEFINITIONS,
            system=SYSTEM_PROMPT,
        )

        return self._execute_react_loop(
            messages=messages,
            response=response,
            file_tools=file_tools,
            project_name=project_name,
        )

    def _execute_react_loop(
        self,
        messages: list,
        response,
        file_tools: BuildTools,
        project_name: str,
    ) -> dict:
        """Execute the ReAct tool-use loop."""
        tool_history: dict = {}
        recent_calls: list[str] = []  # track recent (name, input) for loop detection
        turn = 0
        nudge_count = 0
        turn_extensions_used = 0

        while turn < self.max_turns:
            if response.stop_reason in ("end_turn", "stop"):
                # Model stopped without calling finish — nudge it
                if nudge_count < 2:
                    nudge_count += 1
                    if self.verbose:
                        print(f"[link-units] LLM stopped without finish — nudging (attempt {nudge_count})")

                    # Capture any text the model produced
                    text_content = ""
                    for block in response.content:
                        if hasattr(block, "text") and block.type == "text":
                            text_content += block.text
                    if text_content:
                        messages.append({"role": "assistant", "content": text_content})

                    messages.append({
                        "role": "user",
                        "content": (
                            "You must call finish() with whatever link units you have found. "
                            "Use status='partial' if incomplete. Call finish now."
                        ),
                    })
                    compressed = compress_stale_results(messages, tool_history)
                    truncated = truncate_messages(compressed, max_tokens=MAX_CONTEXT_TOKENS)
                    response = self.llm.complete_with_tools(
                        messages=truncated,
                        tools=DISCOVERY_TOOL_DEFINITIONS,
                        system=SYSTEM_PROMPT,
                    )
                    turn += 1
                    continue
                else:
                    if self.verbose:
                        print(f"[link-units] LLM terminated after {turn + 1} turns without finish")
                    break

            elif response.stop_reason == "tool_use":
                assistant_content = []
                tool_results = []
                finished = False
                finish_result = None

                for block in response.content:
                    if hasattr(block, "text") and block.type == "text":
                        text_entry = {"type": "text", "text": block.text}
                        if getattr(block, "thought", False):
                            text_entry["thought"] = True
                        if getattr(block, "thought_signature", None):
                            text_entry["thought_signature"] = block.thought_signature
                        assistant_content.append(text_entry)
                        if self.verbose and not getattr(block, "thought", False):
                            print(f"[LLM] {block.text}")

                    elif block.type == "tool_use":
                        tool_use_entry = {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        if getattr(block, "thought_signature", None):
                            tool_use_entry["thought_signature"] = block.thought_signature
                        assistant_content.append(tool_use_entry)

                        # Loop detection: skip if same call repeated
                        call_key = json.dumps(
                            {"name": block.name, "input": block.input},
                            sort_keys=True, default=str,
                        )
                        if call_key in recent_calls:
                            result = {
                                "error": (
                                    "You already ran this exact call. "
                                    "Do NOT repeat it. Use the result you already have, "
                                    "or call finish() with your current findings."
                                )
                            }
                            if self.verbose:
                                print(f"[Tool] {block.name} — SKIPPED (duplicate)")
                        else:
                            recent_calls.append(call_key)
                            # Keep a sliding window
                            if len(recent_calls) > 30:
                                recent_calls.pop(0)

                            # Execute tool
                            result = self._execute_tool(
                                block.name, block.input, file_tools,
                            )

                            result = track_tool_result(
                                block.name, block.input, result, tool_history, turn
                            )

                            if self.verbose:
                                print(f"[Tool] {block.name}({json.dumps(block.input, default=str)})")
                                if "error" in result:
                                    print(f"  Error: {str(result['error'])[:200]}")

                        if block.name == "finish":
                            finished = True
                            finish_result = block.input
                        elif block.name == "request_more_turns":
                            if turn_extensions_used < MAX_TURN_EXTENSIONS:
                                self.max_turns += TURNS_EXTENSION
                                turn_extensions_used += 1
                                reason = block.input.get("reason", "")
                                result = {
                                    "granted": TURNS_EXTENSION,
                                    "total_remaining": self.max_turns - turn - 1,
                                    "extensions_used": turn_extensions_used,
                                    "extensions_remaining": MAX_TURN_EXTENSIONS - turn_extensions_used,
                                }
                                if self.verbose:
                                    print(f"[link-units] Granted {TURNS_EXTENSION} extra turns "
                                          f"({turn_extensions_used}/{MAX_TURN_EXTENSIONS}): {reason[:100]}")
                            else:
                                result = {
                                    "error": "Maximum extensions reached. Call finish() now.",
                                    "extensions_used": turn_extensions_used,
                                }

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })

                messages.append({"role": "assistant", "content": assistant_content})

                # Inject turn budget warning
                remaining = self.max_turns - turn - 1
                if remaining in (TURNS_LOW_WARNING, 1) and tool_results:
                    tool_results[-1]["content"] += (
                        f"\n\n[SYSTEM] {remaining} turn(s) left. "
                        f"Call request_more_turns if still making progress, or finish() now."
                    )

                messages.append({"role": "user", "content": tool_results})

                if finished:
                    if self.verbose:
                        status = finish_result.get("status", "unknown") if finish_result else "unknown"
                        print(f"[link-units] Finished after {turn + 1} turns (status={status})")
                    return self._build_output(finish_result, project_name, self.build_dir)

                # Get next response
                if turn < self.max_turns - 1:
                    compressed = compress_stale_results(messages, tool_history)
                    truncated = truncate_messages(compressed, max_tokens=MAX_CONTEXT_TOKENS)

                    if self.verbose and len(truncated) < len(messages):
                        print(f"[Context] Truncated {len(messages) - len(truncated)} messages")

                    response = self.llm.complete_with_tools(
                        messages=truncated,
                        tools=DISCOVERY_TOOL_DEFINITIONS,
                        system=SYSTEM_PROMPT,
                    )
                else:
                    break
            else:
                if self.verbose:
                    print(f"[link-units] Unexpected stop reason: {response.stop_reason}")
                break

            turn += 1

        # Agent didn't call finish — return heuristic results if available
        if self.verbose:
            print("[link-units] Agent did not call finish")
        if self.heuristic_result:
            if self.verbose:
                print("[link-units] Returning heuristic results only")
            return self.heuristic_result
        return {
            "project": project_name,
            "build_system": "unknown",
            "build_dir": str(self.build_dir),
            "link_units": [],
        }

    def _execute_tool(
        self,
        name: str,
        tool_input: dict,
        file_tools: BuildTools,
    ) -> dict:
        """Execute a single tool call and return the result."""
        try:
            if name == "read_file":
                return file_tools.read_file(
                    tool_input["file_path"],
                    max_lines=tool_input.get("max_lines", 200),
                    start_line=tool_input.get("start_line", 1),
                )

            elif name == "list_dir":
                return file_tools.list_dir(
                    tool_input.get("dir_path", "."),
                    pattern=tool_input.get("pattern"),
                )

            elif name == "parse_ninja_targets":
                path_str = tool_input["build_ninja_path"]
                full_path = file_tools._validate_path(path_str)
                return parse_ninja_targets(full_path)

            elif name == "run_ar_t":
                path_str = tool_input["archive_path"]
                full_path = file_tools._validate_path(path_str)
                members = run_ar_t(full_path)
                return {"archive": path_str, "members": members, "count": len(members)}

            elif name == "find_files":
                directory = tool_input["directory"]
                pattern = tool_input["pattern"]
                full_dir = file_tools._validate_path(directory)
                if not full_dir.is_dir():
                    return {"error": f"Not a directory: {directory}"}
                matches = sorted(str(p) for p in full_dir.rglob(pattern))
                truncated = len(matches) > 200
                if truncated:
                    matches = matches[:200]
                return {
                    "directory": directory,
                    "pattern": pattern,
                    "matches": matches,
                    "count": len(matches),
                    "truncated": truncated,
                }

            elif name == "run_command":
                return self._run_command_in_docker(
                    tool_input["command"],
                    workdir=tool_input.get("workdir", "build"),
                )

            elif name == "finish":
                return {"status": "ok", "message": "Discovery complete."}

            elif name == "request_more_turns":
                # Handled by the ReAct loop; this path is only reached via _execute_tool
                return {"acknowledged": True}

            else:
                return {"error": f"Unknown tool: {name}"}

        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _run_command_in_docker(
        self,
        command: str,
        workdir: str = "build",
    ) -> dict:
        """Run an arbitrary shell command inside a Docker container.

        Source and build directories are mounted read-only.
        """
        uid = os.getuid()
        gid = os.getgid()

        work_path = DOCKER_WORKSPACE_BUILD if workdir == "build" else DOCKER_WORKSPACE_SRC

        docker_cmd = [
            "docker", "run", "--rm",
            "-u", f"{uid}:{gid}",
            "-v", f"{self.project_path}:{DOCKER_WORKSPACE_SRC}:ro",
            "-v", f"{self.build_dir}:{DOCKER_WORKSPACE_BUILD}:ro",
            "-w", work_path,
            self.container_image,
            "bash", "-c",
            command,
        ]

        if self.verbose:
            print(f"  [docker] {command[:200]}")

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=DOCKER_COMMAND_TIMEOUT,
            )

            output = result.stdout
            if result.returncode != 0 and result.stderr:
                output += f"\nSTDERR: {result.stderr}"

            # Truncate large output
            if len(output) > 50000:
                output = output[:50000] + "\n... (truncated)"

            return {
                "command": command,
                "workdir": workdir,
                "returncode": result.returncode,
                "output": output,
            }

        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {DOCKER_COMMAND_TIMEOUT}s: {command}"}
        except FileNotFoundError:
            return {"error": "Docker not found. Is Docker installed and in PATH?"}
        except OSError as e:
            return {"error": f"OS error running Docker: {e}"}

    def _build_output(
        self,
        finish_input: dict | None,
        project_name: str,
        build_dir: Path,
    ) -> dict:
        """Build the final link_units.json output from the finish tool input.

        If heuristic_result exists, merges agent-discovered units into it.
        """
        # Start with heuristic results if available
        if self.heuristic_result:
            merged = dict(self.heuristic_result)
            merged["project"] = project_name
        else:
            merged = {
                "project": project_name,
                "build_system": "unknown",
                "build_dir": str(build_dir),
                "link_units": [],
            }

        if not finish_input:
            return merged

        if not self.heuristic_result:
            merged["build_system"] = finish_input.get("build_system", "unknown")

        # Append agent-discovered units
        for unit in finish_input.get("link_units", []):
            merged["link_units"].append({
                "name": unit.get("name", ""),
                "type": unit.get("type", "unknown"),
                "output": unit.get("output", ""),
                "objects": unit.get("objects", []),
                "link_deps": unit.get("link_deps", []),
            })

        return merged
