"""Unified builder with LLM-powered incremental learning."""

import json
import shutil
from pathlib import Path
from typing import Any

from ..llm.base import LLMBackend
from .actions import (
    AutotoolsActions,
    CMakeActions,
    RunCommandAction,
    check_assembly_tool,
    install_packages,
    test_build_script,
)
from .constants import (
    DOCKER_WORKSPACE_BUILD,
    DOCKER_WORKSPACE_SRC,
    MAX_CONTEXT_TOKENS,
    MAX_TURN_EXTENSIONS,
    MAX_TURNS,
    TURNS_EXTENSION,
    TURNS_LOW_WARNING,
)
from .error_analyzer import BuildError, ErrorAnalyzer
from .json_utils import parse_llm_json
from .llm_utils import (
    compress_stale_results,
    estimate_messages_tokens,
    track_tool_result,
    truncate_messages,
)
from .tool_definitions import UNIFIED_TOOL_DEFINITIONS
from .tools import BuildTools


class Builder:
    """Build projects with incremental learning from failures.

    Provides all build tools (CMake + configure/make) to the LLM and lets it
    decide how to build. No build system detection gate.
    """

    def __init__(
        self,
        llm: LLMBackend,
        container_image: str = "llm-summary-builder:latest",
        build_dir: Path | None = None,
        build_scripts_dir: Path | None = None,
        max_retries: int = 3,
        enable_lto: bool = True,
        prefer_static: bool = True,
        generate_ir: bool = True,
        verbose: bool = False,
        log_file: str | None = None,
        ccache_dir: Path | None = None,
    ):
        self.llm = llm
        self.container_image = container_image
        self.build_dir = build_dir
        self.build_scripts_dir = Path(build_scripts_dir) if build_scripts_dir else None
        self.max_retries = max_retries
        self.enable_lto = enable_lto
        self.prefer_static = prefer_static
        self.generate_ir = generate_ir
        self.verbose = verbose
        self.log_file = log_file
        self.error_analyzer = ErrorAnalyzer(llm, verbose, log_file=log_file)
        self.ccache_dir = ccache_dir

    def _get_unavoidable_asm_path(self, project_name: str) -> Path | None:
        """Get path to unavoidable_asm.json for a project."""
        if not self.build_scripts_dir:
            return None
        return self.build_scripts_dir / project_name.lower() / "unavoidable_asm.json"

    def _load_prior_config(self, project_name: str) -> dict | None:
        """Load prior build config from build-scripts/<project>/config.json."""
        if not self.build_scripts_dir:
            return None
        config_path = self.build_scripts_dir / project_name.lower() / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path) as f:
                config: dict = json.load(f)
            if self.verbose:
                print(f"[Prior Config] Loaded from {config_path}")
            return config
        except (json.JSONDecodeError, OSError) as e:
            if self.verbose:
                print(f"[Prior Config] Failed to load {config_path}: {e}")
            return None

    def _format_prior_config_message(self, project_name: str, config: dict) -> str:
        """Format a user message that includes prior build configuration."""
        build_system = config.get("build_system", "unknown")
        parts = [
            f"Build this project: {project_name}",
            "",
            'Note: All file paths must be relative to the project root. '
            'Build directory is at "build/".',
            "",
            "## Prior build configuration",
            "",
            "A previous successful build used the following configuration. "
            "Use this as a starting point — deviate if needed.",
            "",
            f"Build system: {build_system}",
        ]

        # CMake flags
        cmake_flags = config.get("cmake_flags")
        if cmake_flags:
            parts.append(f"CMake flags: {' '.join(cmake_flags)}")

        # Configure flags
        configure_flags = config.get("configure_flags")
        if configure_flags:
            parts.append(f"Configure flags: {' '.join(configure_flags)}")

        # use_build_dir
        if "use_build_dir" in config:
            parts.append(f"use_build_dir: {config['use_build_dir']}")

        # Dependencies
        dependencies = config.get("dependencies")
        if dependencies:
            parts.append(f"Dependencies to install: {', '.join(dependencies)}")

        # Build script
        build_script = config.get("build_script")
        if build_script:
            parts.extend([
                "",
                "Prior build script:",
                "```",
                build_script.rstrip(),
                "```",
                "",
                "Use test_build_script to validate the script (adjust if needed).",
            ])

        return "\n".join(parts)

    def learn_and_build(
        self, project_path: Path
    ) -> dict[str, Any]:
        """
        Learn how to build a project through iterative attempts.

        Returns a dict with:
        - success: bool
        - flags: list[str]
        - build_system_used: str ("cmake", "configure_make", "make")
        - attempts: int
        - build_log: str
        - error_messages: list[str]
        - use_build_dir: bool
        """
        project_path = Path(project_path).resolve()

        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Clean build directory to ensure fresh build
        build_dir = Path(self.build_dir) if self.build_dir else project_path / "build"
        if build_dir.exists():
            if self.verbose:
                print(f"[Cleanup] Removing existing build directory: {build_dir}")
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Get initial configuration from LLM
        if self.verbose:
            print("\n[1/3] Analyzing project with LLM...")

        result = self._get_initial_config_with_tools(project_path)

        # Handle both list[str] (simple mode) and dict (ReAct mode with metadata)
        if isinstance(result, dict):
            flags = result.get("flags", self._get_default_cmake_config())
            build_system_used = result.get("build_system_used", "unknown")
            react_build_succeeded = result.get("build_succeeded", False)
            react_terminated = result.get("react_terminated", False)
            use_build_dir = result.get("use_build_dir", True)

            if react_build_succeeded:
                if self.verbose:
                    print("\n[ReAct] Build already succeeded in exploration phase!")
                    print(f"[3/3] Build successful after {result.get('attempts', 1)} turns!")

                return {
                    "success": True,
                    "flags": flags,
                    "build_system_used": build_system_used,
                    "attempts": 1,
                    "build_log": "Build succeeded during ReAct exploration",
                    "error_messages": [],
                    "use_build_dir": use_build_dir,
                    "dependencies": result.get("dependencies", []),
                    "build_script": result.get("build_script"),
                }

            if react_terminated:
                if self.verbose:
                    print("\n[ReAct] Build terminated - LLM identified unresolvable blockers")
                    print("[FAILED] Cannot proceed with available tools")

                return {
                    "success": False,
                    "flags": flags,
                    "build_system_used": build_system_used,
                    "attempts": 1,
                    "build_log": "ReAct loop terminated without successful build",
                    "error_messages": [
                        "LLM identified blockers that cannot be "
                        "resolved with available tools "
                        "(e.g., missing dependencies)"
                    ],
                    "use_build_dir": use_build_dir,
                    "dependencies": result.get("dependencies", []),
                }

            # ReAct loop exhausted max turns without completing
            if self.verbose:
                print("\n[ReAct] Loop exhausted max turns without completing build")
                print("[FAILED] Agent did not finish the build workflow in time")

            return {
                "success": False,
                "flags": flags,
                "build_system_used": build_system_used,
                "attempts": 1,
                "build_log": "ReAct loop exhausted max turns without completing build",
                "error_messages": ["Agent ran out of turns before completing the build workflow"],
                "use_build_dir": use_build_dir,
                "dependencies": result.get("dependencies", []),
            }
        else:
            # Simple mode returned a list of cmake flags
            flags = result
            build_system_used = "cmake"
            use_build_dir = True
            react_build_succeeded = False

        if self.verbose:
            print(f"[Initial Config] {len(flags)} flags:")
            for flag in flags:
                print(f"  {flag}")

        # Attempt build with retries (fallback for simple mode)
        build_log: list[str] = []
        error_messages: list[str] = []
        attempts = 0

        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            if self.verbose:
                print(f"\n[2/3] Build attempt {attempt}/{self.max_retries}...")

            try:
                log = self._attempt_cmake_build(project_path, flags)
                build_log.append(log)

                if self.verbose:
                    print(f"\n[3/3] Build successful after {attempt} attempts!")

                return {
                    "success": True,
                    "flags": flags,
                    "build_system_used": "cmake",
                    "attempts": attempts,
                    "build_log": "\n\n".join(build_log),
                    "error_messages": error_messages,
                    "use_build_dir": True,
                }

            except BuildError as e:
                error_msg = str(e)
                error_messages.append(error_msg)
                build_log.append(f"Attempt {attempt} failed:\n{error_msg}")

                if self.verbose:
                    print(f"[ERROR] Attempt {attempt} failed:")
                    print(f"  {error_msg[:500]}...")

                if attempt >= self.max_retries:
                    if self.verbose:
                        print(f"\n[FAILED] Max retries ({self.max_retries}) reached.")
                    break

                if self.verbose:
                    print("\n[LLM] Analyzing error to adjust configuration...")

                is_cmake_error = "CMake Error" in error_msg or "cmake" in error_msg.lower()

                file_tools = BuildTools(project_path, build_dir)
                analysis = self.error_analyzer.analyze_error_with_tools(
                    error_msg,
                    flags,
                    project_path,
                    file_tools,
                    is_cmake_error=is_cmake_error,
                )

                if self.verbose:
                    print(f"[Analysis] {analysis['diagnosis']}")
                    print(f"[Confidence] {analysis['confidence']}")

                flags = self._apply_suggestions(flags, analysis)

                if self.verbose:
                    print(f"[Updated Config] {len(flags)} flags")

        return {
            "success": False,
            "flags": flags,
            "build_system_used": build_system_used,
            "attempts": attempts,
            "build_log": "\n\n".join(build_log),
            "error_messages": error_messages,
            "use_build_dir": use_build_dir,
        }

    def _get_initial_config_with_tools(
        self, project_path: Path
    ) -> list[str] | dict:
        """
        Unified approach: provide all build tools and let LLM explore and build.

        Returns:
            - list[str]: flags (simple mode)
            - dict: Full result including flags and build status (ReAct mode)
        """
        project_name = project_path.name
        build_dir = Path(self.build_dir) if self.build_dir else project_path / "build"

        # Ensure ccache host directory exists
        if self.ccache_dir:
            self.ccache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tools for both build systems
        file_tools = BuildTools(project_path, build_dir)
        unavoidable_asm_path = self._get_unavoidable_asm_path(project_name)
        cmake_actions = CMakeActions(
            project_path, self.build_dir, self.container_image,
            unavoidable_asm_path=unavoidable_asm_path, verbose=self.verbose,
            ccache_dir=self.ccache_dir,
        )
        autotools_actions = AutotoolsActions(
            project_path, self.build_dir, self.container_image,
            unavoidable_asm_path=unavoidable_asm_path, verbose=self.verbose,
            ccache_dir=self.ccache_dir,
        )
        run_command_action = RunCommandAction(
            project_path, self.build_dir, self.container_image,
            verbose=self.verbose, ccache_dir=self.ccache_dir,
        )

        system = (
            "You are a build configuration expert. "
            "Build this project iteratively using "
            "the available tools.\n\n"
            f"**Turn Budget:** You have {MAX_TURNS} tool-use "
            "turns to complete the build. Be efficient — "
            "minimize exploratory reads and move to building "
            "quickly. If you recognize the project or build "
            "system, skip unnecessary exploration. You will "
            "be warned when turns are running low. If you "
            "need more turns (e.g., complex build with many "
            "dependencies), call the request_more_turns tool "
            "with a reason.\n\n"
            "**Step 1 - Explore the project root** to "
            "determine the build system:\n"
            "- Use list_dir and read_file to examine the "
            "project structure\n"
            "- Keep exploration minimal — 2-3 reads max "
            "before starting the build\n\n"
            "**Step 2 - Choose the appropriate build "
            "approach:**\n"
            "- If `CMakeLists.txt` exists → prefer "
            "cmake_configure + cmake_build (generates "
            "compile_commands.json natively)\n"
            "- If `configure` or `configure.ac` exists → "
            "use run_configure + make_build; run autoreconf "
            "first if only configure.ac\n"
            "- If only `Makefile` exists → use make_build "
            "directly with use_build_dir=false\n"
            "- For Meson, Bazel, SCons, or custom build "
            "systems → use run_command to explore and build\n"
            "- For simple/straightforward CMake projects, "
            "you may return JSON configuration instead of "
            "using tools\n\n"

            "**Simple mode (for straightforward CMake "
            "projects):**\n"
            "Return JSON configuration directly:\n"
            "{{\n"
            '  "cmake_flags": '
            '["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ...],\n'
            '  "reasoning": "...",\n'
            '  "dependencies": [...]\n'
            "}}\n\n"
            "**Available Tools:**\n"
            "- read_file: Read project files and build "
            "artifacts\n"
            "- list_dir: Explore project structure and "
            "build directory\n"
            "- cmake_configure: Run cmake configure with "
            "flags\n"
            "- cmake_build: Run ninja build after cmake "
            "configure\n"
            "- bootstrap: Run bootstrap/autogen.sh script\n"
            "- autoreconf: Run autoreconf -fi to regenerate "
            "configure script\n"
            "- run_configure: Run ./configure with flags "
            "(env vars default to clang-18 + LTO, but can "
            "be overridden)\n"
            "- make_build: Run bear -- make to build and "
            "capture compile commands\n"
            "- make_clean: Run make clean\n"
            "- make_distclean: Run make distclean\n"
            "- run_command: Run arbitrary shell commands in "
            "Docker (for Meson, Bazel, SCons, custom build "
            "systems)\n"
            "- test_build_script: Test a build script from "
            "scratch in a clean temp dir (verifies "
            "compile_commands.json)\n"
            "- check_assembly: Explicit assembly check after "
            "run_command builds\n"
            "- install_packages: Install system packages "
            "(apt) when build fails due to missing "
            "headers/libraries\n"
            "- request_more_turns: Request additional turns "
            "if running low and still making progress\n"
            "- finish: Signal completion (MUST call this "
            "when done)\n\n"
            "**IMPORTANT**: All file/directory paths in "
            "tools must be RELATIVE to project root (e.g., "
            '".", "cmake/", "src/config.h", '
            '"build/compile_commands.json"). The build '
            'directory is accessible at "build/". Absolute '
            "paths are not allowed.\n\n"
            "**Build Requirements (mandatory):**\n"
            "- Generate compile_commands.json (cmake: "
            "CMAKE_EXPORT_COMPILE_COMMANDS=ON; make: via "
            "bear)\n\n"
            "**Build Preferences (use these by default, but "
            "fall back if the project doesn't support "
            "them):**\n"
            "- Disable SIMD/hardware optimizations to "
            "minimize assembly code (preferred but not "
            "required - do NOT spend excessive turns on "
            "this)\n"
            "- Prefer Clang 18 (cmake: "
            "CMAKE_C_COMPILER=clang-18; configure/make: "
            "auto-injected env vars). If the project fails "
            "to build with clang, fall back to gcc.\n"
            "- Prefer LLVM LTO (cmake: "
            "CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON; "
            "configure/make: auto-injected). If LTO causes "
            "build failures, disable it.\n"
            "- Prefer static libraries only (cmake: "
            "BUILD_SHARED_LIBS=OFF; configure: "
            "--disable-shared --enable-static). If the "
            "project requires shared libraries, that's "
            "fine.\n"
            "- Prefer LLVM IR generation (cmake: "
            "CMAKE_C_FLAGS='-g -flto=full -save-temps=obj';"
            " configure/make: auto-injected). Only "
            "applicable when using clang with LTO. The -g "
            "flag is required so .bc files retain debug info"
            " for downstream analysis.\n\n"
            "**Assembly Verification:**\n"
            "After each successful build (cmake_build or "
            "make_build), an assembly check runs "
            "automatically. If assembly is detected, you "
            "may try ONE round of different flags to reduce "
            "it, but a successful build with assembly is "
            "acceptable. Do not waste turns trying to "
            "eliminate all assembly.\n\n"
            "**Tool ordering rules:**\n"
            "- cmake_configure can be called freely (cmake "
            "handles reconfigure)\n"
            "- cmake_build requires a prior "
            "cmake_configure\n"
            "- run_configure requires make_distclean first "
            "if you already configured (to reconfigure with "
            "new flags)\n"
            "- make_build requires a prior run_configure "
            "(unless Makefile-only project with "
            "use_build_dir=false)\n\n"
            "**Normal workflow:**\n"
            "explore (list_dir/read_file) → configure "
            "(cmake_configure or run_configure) → build "
            "(cmake_build or make_build) → finish\n\n"
            "**Workflow for Meson, Bazel, SCons, or custom "
            "build systems:**\n"
            "1. Use run_command to explore the project and "
            "try building (trial and error)\n"
            "2. After figuring out the build, distill "
            "working commands into a clean script\n"
            "3. Call test_build_script(script) to verify it "
            "works from scratch\n"
            "   - The test runs in a fresh build dir and "
            "REQUIRES compile_commands.json to be "
            "generated\n"
            "   - If it fails, adjust the script and "
            "retry\n"
            "4. Only call finish() after test_build_script "
            "succeeds\n"
            "5. Include the validated build_script in "
            "finish(build_script=...)\n\n"
            "**Assembly minimization (optional, best-effort):**\n"
            "After a successful build, if assembly is "
            "detected, you may try ONE iteration to reduce "
            "it:\n"
            "- CMake: cmake_configure (new flags) → "
            "cmake_build\n"
            "- Make: make_distclean → run_configure (new "
            "flags) → make_build\n"
            "- Custom: adjust script → test_build_script\n"
            "If the first attempt doesn't help, accept the "
            "result and call finish. Do NOT iterate "
            "multiple times trying to eliminate assembly.\n\n"
            "**Installing missing packages:**\n"
            "- If a build fails due to missing headers or "
            "libraries (e.g., zlib.h not found), use "
            "install_packages to install the required dev "
            "package (e.g., zlib1g-dev)\n"
            "- After installing, retry the build (you may "
            "need to re-run cmake_configure or "
            "run_configure)\n"
            "- When calling finish, report which installed "
            "packages are actual build dependencies in the "
            "'dependencies' field\n\n"
            "**CRITICAL - When to STOP:**\n"
            "- Once build succeeds (with or without "
            "assembly), call finish(status=\"success\", "
            'summary="...", dependencies=[...])\n'
            "- If you've exhausted all reasonable options, "
            'call finish(status="failure", '
            'summary="...")\n\n'
            f"You are working on: {project_name}\n"
            "If you recognize this project, leverage your "
            "knowledge of its typical build requirements."
        )

        # Build the initial user message, incorporating prior config if available
        prior_config = self._load_prior_config(project_name)
        if prior_config:
            user_content = self._format_prior_config_message(project_name, prior_config)
        else:
            user_content = (
                f"Build this project: {project_name}\n\n"
                "Note: All file paths must be relative to the project root. "
                'Build directory is at "build/".\n\n'
                "Explore the project root first to determine the build system, "
                "then proceed with the build."
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

        if self.verbose:
            print("[LLM] Requesting initial configuration (unified mode)...")
            print(f"[LLM] Project: {project_name}")
            print(
                "[LLM] Available tools: read_file, "
                "list_dir, cmake_configure, cmake_build, "
                "bootstrap, autoreconf, run_configure, "
                "make_build, make_clean, make_distclean, "
                "run_command, test_build_script, "
                "check_assembly, install_packages, "
                "request_more_turns, finish"
            )

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write("BUILD-LEARN UNIFIED MODE - INITIAL REQUEST\n")
                f.write(f"Project: {project_name}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"SYSTEM PROMPT:\n{system}\n\n")
                f.write(f"USER MESSAGE:\n{messages[0]['content']}\n\n")
                f.write(f"TOOLS:\n{json.dumps(UNIFIED_TOOL_DEFINITIONS, indent=2)}\n\n")

        # First turn
        response = self.llm.complete_with_tools(
            messages=messages,
            tools=UNIFIED_TOOL_DEFINITIONS,
            system=system,
        )

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("RESPONSE:\n")
                f.write(f"Stop reason: {response.stop_reason}\n")
                for i, block in enumerate(response.content):
                    if hasattr(block, 'text'):
                        f.write(f"[Block {i}] Text: {block.text}\n")
                    elif hasattr(block, 'name'):
                        f.write(f"[Block {i}] Tool use: {block.name}({block.input})\n")
                f.write("\n")

        # OpenAI format returns "stop", Anthropic returns "end_turn"
        if response.stop_reason in ("end_turn", "stop"):
            # Simple mode: LLM returned text (likely JSON config)
            if self.verbose:
                print("[Mode] Simple workflow (LLM provided config directly)")

            text = self._extract_text(response)
            config = self._parse_config_response(text)

            if config:
                return config
            else:
                if self.verbose:
                    print("[Mode] Failed to parse JSON, using default config")
                return self._get_default_cmake_config()

        elif response.stop_reason == "tool_use":
            # ReAct mode
            if self.verbose:
                print("[Mode] ReAct workflow (LLM exploring with tools)")

            result = self._execute_react_loop(
                messages, response, file_tools, cmake_actions, autotools_actions,
                run_command_action, system
            )
            return result

        else:
            if self.verbose:
                print(f"[Mode] Unexpected stop_reason: {response.stop_reason}, using default")
            return self._get_default_cmake_config()

    def _parse_config_response(self, response: str) -> list[str]:
        """Parse LLM response to extract CMake configuration."""
        result = parse_llm_json(response, default_response={}, verbose=self.verbose)

        if result:
            flags: list[str] = result.get("cmake_flags", [])

            if self.verbose and result.get("reasoning"):
                print(f"[LLM Reasoning] {result['reasoning'][:200]}...")

            if result.get("dependencies"):
                print(f"[Dependencies] LLM identified {len(result['dependencies'])} dependencies:")
                for dep in result["dependencies"]:
                    if isinstance(dep, str):
                        print(f"  - {dep}")
                    elif isinstance(dep, dict):
                        pkg = dep.get("package", "unknown")
                        reason = dep.get("reason", "")
                        print(f"  - {pkg}: {reason}")
                    else:
                        print(f"  - {dep}")
                print("[Note] Ensure these are installed in the Docker image or build may fail")

            if self.verbose and result.get("potential_issues"):
                print("[Potential Issues]")
                for issue in result["potential_issues"]:
                    print(f"  - {issue}")

            return flags
        else:
            return self._get_default_cmake_config()

    def _extract_text(self, response) -> str:
        """Extract all text blocks from response."""
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text

    def _execute_react_loop(
        self,
        messages: list,
        initial_response,
        file_tools: BuildTools,
        cmake_actions: CMakeActions,
        autotools_actions: AutotoolsActions,
        run_command_action: RunCommandAction,
        system: str,
    ) -> dict:
        """
        Unified ReAct-style build loop where LLM uses tools iteratively.

        Returns:
            Dict with 'success', 'flags', 'build_system_used', 'attempts',
            'configure_succeeded', 'build_succeeded', 'react_terminated', 'use_build_dir',
            'build_script'
        """
        max_turns = MAX_TURNS
        turn_extensions_used = 0
        configure_succeeded = False
        build_succeeded = False
        react_terminated = False
        finished = False
        finish_status = None
        finish_summary = None
        final_flags = []
        build_system_used = "unknown"
        use_build_dir = True
        build_script: str | None = None
        tool_history: dict[str, Any] = {}
        installed_packages: set[str] = set()
        dependencies: list[str] = []

        # State machine for enforcing tool ordering
        # cmake: cmake_configure can be called freely; cmake_build requires prior cmake_configure
        # make:  run_configure requires make_distclean if already configured;
        #        make_build requires prior run_configure (or Makefile-only project)
        cmake_configured = False
        make_configured = False  # run_configure succeeded (Makefile exists)

        response = initial_response
        turn = 0

        while turn < max_turns:
            if response.stop_reason in ("end_turn", "stop"):
                if not build_succeeded:
                    react_terminated = True
                    if self.verbose:
                        print(
                            "[ReAct] LLM terminated without "
                            f"successful build after {turn + 1}"
                            " turns"
                        )
                        print(
                            "[ReAct] This typically means "
                            "the LLM identified blockers "
                            "it cannot resolve"
                        )
                else:
                    if self.verbose:
                        print(f"[ReAct] LLM finished after {turn + 1} turns")
                break

            elif response.stop_reason == "tool_use":
                assistant_content = []
                tool_results = []

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

                        # State guard: enforce tool ordering
                        result = self._check_state_guard(
                            block.name, block.input,
                            cmake_configured, make_configured,
                            build_succeeded,
                            build_system_used=build_system_used,
                        )

                        if result is None:
                            # No guard violation, execute normally
                            result = self._execute_tool_safe(
                                file_tools, cmake_actions, autotools_actions,
                                block.name, block.input,
                                run_command_action=run_command_action,
                            )

                        result = track_tool_result(
                            block.name, block.input, result, tool_history, turn
                        )

                        if self.verbose:
                            print(f"[Tool] {block.name}({json.dumps(block.input)})")
                            if "success" in result:
                                print(f"  Success: {result['success']}")
                            if "error" in result and result["error"]:
                                print(f"  Error: {result['error'][:200]}")

                        # Update state machine on success
                        if block.name == "cmake_configure" and result.get("success"):
                            cmake_configured = True
                            configure_succeeded = True
                            build_system_used = "cmake"
                            final_flags = block.input.get("cmake_flags", [])
                        elif block.name == "cmake_build" and result.get("success"):
                            build_succeeded = True
                            build_system_used = "cmake"
                        elif block.name == "run_configure" and result.get("success"):
                            # Only advance state if Makefile was actually generated
                            # (e.g., ./configure --help succeeds but produces nothing)
                            _ubd = block.input.get("use_build_dir", True)
                            if _ubd:
                                _mf_dir = (
                                    autotools_actions.build_dir
                                    or (autotools_actions.project_path / "build")
                                )
                            else:
                                _mf_dir = autotools_actions.project_path
                            if (_mf_dir / "Makefile").exists():
                                make_configured = True
                                configure_succeeded = True
                                build_system_used = "configure_make"
                                final_flags = block.input.get("configure_flags", [])
                                use_build_dir = _ubd
                            elif self.verbose:
                                print(
                                    "[State] run_configure returned "
                                    "success but no Makefile at "
                                    f"{_mf_dir}"
                                )
                        elif block.name == "make_build" and result.get("success"):
                            build_succeeded = True
                            if build_system_used == "unknown":
                                build_system_used = "make"
                            if block.input.get("use_build_dir") is not None:
                                use_build_dir = block.input.get("use_build_dir", True)
                        elif block.name == "make_distclean":
                            # Reset make state regardless of success (distclean is best-effort)
                            make_configured = False
                            build_succeeded = False
                        elif block.name == "make_clean":
                            build_succeeded = False
                        elif block.name == "run_command" and result.get("success"):
                            # Check if run_command produced compile_commands.json
                            cc_in_build = (
                                (Path(self.build_dir) / "compile_commands.json")
                                if self.build_dir else None
                            )
                            cc_in_src = cmake_actions.project_path / "compile_commands.json"
                            cc_exists = (
                                (cc_in_build and cc_in_build.exists())
                                or cc_in_src.exists()
                            )
                            if cc_exists:
                                build_succeeded = True
                                if build_system_used == "unknown":
                                    build_system_used = "custom"
                                if self.verbose:
                                    print(
                                        "[State] run_command produced "
                                        "compile_commands.json — "
                                        "build validated"
                                    )
                        elif block.name == "test_build_script" and result.get("success"):
                            build_succeeded = True
                            if self.verbose:
                                print("[ReAct] test_build_script succeeded — build validated")
                        elif block.name == "install_packages" and result.get("success"):
                            new_image = result.get("new_image")
                            if new_image:
                                cmake_actions.container_image = new_image
                                autotools_actions.container_image = new_image
                                run_command_action.container_image = new_image
                                installed_packages.update(block.input.get("packages", []))
                                if self.verbose:
                                    print(f"[ReAct] Updated container image to {new_image}")
                        elif block.name == "request_more_turns":
                            if turn_extensions_used < MAX_TURN_EXTENSIONS:
                                max_turns += TURNS_EXTENSION
                                turn_extensions_used += 1
                                reason = block.input.get("reason", "")
                                result = {
                                    "granted": True,
                                    "extra_turns": TURNS_EXTENSION,
                                    "new_max_turns": max_turns,
                                    "remaining_turns": max_turns - turn - 1,
                                    "extensions_remaining": (
                                        MAX_TURN_EXTENSIONS - turn_extensions_used
                                    ),
                                }
                                if self.verbose:
                                    print(
                                        f"[ReAct] Granted {TURNS_EXTENSION} "
                                        f"extra turns (reason: {reason})"
                                    )
                                    remaining = max_turns - turn - 1
                                    print(
                                        f"[ReAct] New max: {max_turns}, "
                                        f"remaining: {remaining}"
                                    )
                            else:
                                result = {
                                    "granted": False,
                                    "error": (
                                        f"Maximum extensions ({MAX_TURN_EXTENSIONS}) "
                                        "already used. Wrap up and call finish."
                                    ),
                                    "remaining_turns": max_turns - turn - 1,
                                }
                                if self.verbose:
                                    print("[ReAct] Denied turn extension — max extensions reached")
                        elif block.name == "finish" and not result.get("error"):
                            finished = True
                            finish_status = block.input.get("status")
                            finish_summary = block.input.get("summary")
                            # Capture build_script if provided
                            if block.input.get("build_script"):
                                build_script = block.input["build_script"]
                            # Agent-reported build system overrides auto-detected
                            if block.input.get("build_system"):
                                build_system_used = block.input["build_system"]
                            # Cross-validate: only keep deps that were actually installed
                            reported_deps = block.input.get("dependencies", [])
                            dependencies = [
                                d for d in reported_deps if d in installed_packages
                            ]
                            if self.verbose:
                                print(f"[ReAct] LLM called finish: status={finish_status}")
                                print(f"[ReAct] Summary: {finish_summary}")
                                if block.input.get("build_system"):
                                    print(f"[ReAct] Build system: {build_system_used}")
                                if reported_deps:
                                    print(f"[ReAct] Reported dependencies: {reported_deps}")
                                    print(f"[ReAct] Validated dependencies: {dependencies}")
                                if build_script:
                                    bs_len = len(build_script)
                                    print(
                                        "[ReAct] Build script "
                                        f"provided ({bs_len} chars)"
                                    )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                messages.append({"role": "assistant", "content": assistant_content})

                # Inject turn budget warning when running low
                # Appended to last tool result's content to stay compatible with
                # all backends (OpenAI-compatible APIs reject mixed content types)
                remaining_turns = max_turns - turn - 1
                warning = None
                if remaining_turns == TURNS_LOW_WARNING and not build_succeeded:
                    warning = (
                        f"\n\n[SYSTEM WARNING] You have {remaining_turns} turns remaining. "
                        "Wrap up exploration and focus on completing the build. "
                        "If you need more turns, call request_more_turns now. "
                        "You MUST call finish() before running out of turns."
                    )
                    if self.verbose:
                        print(
                            "[ReAct] Injected low-turn warning "
                            f"({remaining_turns} turns left)"
                        )
                elif remaining_turns == 1 and not build_succeeded:
                    warning = (
                        "\n\n[SYSTEM WARNING] This is your LAST turn. "
                        "Call finish(status='failure', "
                        "summary='...') now to report "
                        "what you accomplished."
                    )
                    if self.verbose:
                        print("[ReAct] Injected final-turn warning")

                if warning and tool_results:
                    last = tool_results[-1]
                    last["content"] = last.get("content", "") + warning

                messages.append({"role": "user", "content": tool_results})

                if finished:
                    if self.verbose:
                        print(f"[ReAct] Finished after {turn + 1} turns")
                    break

                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"REACT TURN {turn + 1} - TOOL RESULTS\n")
                        f.write(f"{'='*80}\n\n")
                        for tr in tool_results:
                            f.write(f"Tool result for {tr.get('tool_use_id')}:\n")
                            f.write(f"{tr.get('content')}\n\n")

                # Get next response
                if turn < max_turns - 1:
                    compressed_messages = compress_stale_results(messages, tool_history)
                    truncated_messages = truncate_messages(
                        compressed_messages,
                        max_tokens=MAX_CONTEXT_TOKENS,
                    )

                    if self.verbose and len(truncated_messages) < len(messages):
                        n = len(messages) - len(truncated_messages)
                        print(f"[Context] Truncated {n} messages")
                        est = estimate_messages_tokens(truncated_messages)
                        print(f"[Context] Estimated tokens: {est}")

                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"REACT TURN {turn + 1} - LLM REQUEST\n")
                            f.write(f"{'='*80}\n\n")
                            f.write(f"Messages count: {len(truncated_messages)}\n")
                            est = estimate_messages_tokens(
                                truncated_messages
                            )
                            f.write(f"Estimated tokens: {est}\n")
                            f.write(f"System prompt length: {len(system)} chars\n")
                            f.write(f"Tools: {len(UNIFIED_TOOL_DEFINITIONS)} available\n\n")
                            f.write("MESSAGES:\n")
                            for i, msg in enumerate(truncated_messages):
                                f.write(f"[Message {i}] Role: {msg.get('role')}\n")
                                content = msg.get('content', '')
                                if isinstance(content, str):
                                    preview = (
                                        content[:200] + '...'
                                        if len(content) > 200
                                        else content
                                    )
                                    f.write(f"  Content (preview): {preview}\n")
                                elif isinstance(content, list):
                                    f.write(f"  Content blocks: {len(content)}\n")
                                    for j, block in enumerate(content[:3]):
                                        if isinstance(block, dict):
                                            block_type = block.get('type', 'unknown')
                                            f.write(f"    Block {j}: type={block_type}\n")
                            f.write("\n")

                    response = self.llm.complete_with_tools(
                        messages=truncated_messages,
                        tools=UNIFIED_TOOL_DEFINITIONS,
                        system=system,
                    )

                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write("LLM RESPONSE:\n")
                            f.write(f"Stop reason: {response.stop_reason}\n")
                            for i, block in enumerate(response.content):
                                if hasattr(block, 'text'):
                                    f.write(f"[Block {i}] Text: {block.text}\n")
                                elif hasattr(block, 'name'):
                                    f.write(f"[Block {i}] Tool use: {block.name}({block.input})\n")
                            f.write("\n")
                else:
                    break

            else:
                if self.verbose:
                    print(f"[ReAct] Unexpected stop reason: {response.stop_reason}")
                break

            turn += 1

        if not final_flags:
            final_flags = self._get_default_cmake_config()

        return {
            "success": build_succeeded,
            "flags": final_flags,
            "build_system_used": build_system_used,
            "attempts": turn + 1,
            "configure_succeeded": configure_succeeded,
            "build_succeeded": build_succeeded,
            "react_terminated": react_terminated,
            "use_build_dir": use_build_dir,
            "dependencies": dependencies,
            "build_script": build_script,
        }

    def _check_state_guard(
        self,
        tool_name: str,
        tool_input: dict,
        cmake_configured: bool,
        make_configured: bool,
        build_succeeded: bool,
        build_system_used: str = "unknown",
    ) -> dict | None:
        """Check state machine constraints before executing a tool.

        Returns None if the tool is allowed, or an error dict if blocked.

        Rules:
        - cmake_configure: always allowed (cmake handles reconfigure)
        - cmake_build: requires cmake_configure first
        - run_configure: requires make_distclean first if already configured
        - make_build: requires run_configure first (unless Makefile-only with use_build_dir=false)
        - make_clean/make_distclean: always allowed
        - finish(status=success): requires build_succeeded
          (cmake_build, make_build, or test_build_script must have succeeded)
        """
        if tool_name == "cmake_build" and not cmake_configured:
            if self.verbose:
                print("[Guard] Blocked cmake_build — cmake_configure required first")
            return {"error": "Run cmake_configure before cmake_build."}

        if tool_name == "run_configure" and make_configured:
            # Allow --help even when already configured (it's read-only)
            flags = tool_input.get("configure_flags", [])
            if "--help" in flags:
                return None
            if self.verbose:
                print("[Guard] Blocked run_configure — make_distclean required first")
            msg = "Already configured."
            if build_succeeded:
                msg += " Previous build succeeded."
            msg += " Run make_distclean first, then run_configure with new flags."
            return {"error": msg}

        if tool_name == "make_build" and not make_configured:
            # Allow make_build without configure for Makefile-only projects
            if not tool_input.get("use_build_dir", True):
                return None
            if self.verbose:
                print("[Guard] Blocked make_build — run_configure required first")
            return {"error": "Run run_configure before make_build."}

        if tool_name == "finish" and tool_input.get("status") == "success" and not build_succeeded:
            if build_system_used == "cmake":
                msg = "Run cmake_build before finishing with success."
            elif build_system_used in ("configure_make", "make"):
                msg = "Run make_build before finishing with success."
            else:
                msg = (
                    "No validated build. Call test_build_script with your build "
                    "commands to verify the build is reproducible and generates "
                    "compile_commands.json, then call finish."
                )
            if self.verbose:
                print("[Guard] Blocked finish(success) — build not validated")
            return {"error": msg}

        return None

    def _execute_tool_safe(
        self,
        file_tools: BuildTools,
        cmake_actions: CMakeActions,
        autotools_actions: AutotoolsActions,
        tool_name: str,
        tool_input: dict,
        run_command_action: RunCommandAction | None = None,
    ) -> dict:
        """Execute tool with security checks.

        Routes to CMake, autotools, and custom build actions.
        """
        try:
            # File tools
            if tool_name == "read_file":
                return file_tools.read_file(
                    file_path=tool_input.get("file_path", ""),
                    max_lines=tool_input.get("max_lines", 200),
                    start_line=tool_input.get("start_line", 1),
                )
            elif tool_name == "list_dir":
                return file_tools.list_dir(
                    dir_path=tool_input.get("dir_path", "."),
                    pattern=tool_input.get("pattern"),
                )
            # CMake tools
            elif tool_name == "cmake_configure":
                return cmake_actions.cmake_configure(
                    cmake_flags=tool_input.get("cmake_flags", [])
                )
            elif tool_name == "cmake_build":
                return cmake_actions.cmake_build()
            # Configure/Make tools
            elif tool_name == "bootstrap":
                return autotools_actions.bootstrap(
                    script_path=tool_input.get("script_path", "bootstrap"),
                )
            elif tool_name == "autoreconf":
                return autotools_actions.autoreconf()
            elif tool_name == "run_configure":
                return autotools_actions.run_configure(
                    configure_flags=tool_input.get("configure_flags", []),
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "make_build":
                return autotools_actions.make_build(
                    make_target=tool_input.get("make_target", ""),
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "make_clean":
                return autotools_actions.make_clean(
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "make_distclean":
                return autotools_actions.make_distclean(
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            # Arbitrary build system tools
            elif tool_name == "run_command":
                if run_command_action is None:
                    return {"error": "run_command not available"}
                return run_command_action.run_command(
                    command=tool_input.get("command", ""),
                    workdir=tool_input.get("workdir", "src"),
                )
            elif tool_name == "test_build_script":
                return test_build_script(
                    script_content=tool_input.get("script", ""),
                    project_path=cmake_actions.project_path,
                    build_dir=cmake_actions.build_dir,
                    container_image=cmake_actions.container_image,
                    unavoidable_asm_path=cmake_actions.unavoidable_asm_path,
                    verbose=self.verbose,
                    ccache_dir=self.ccache_dir,
                )
            elif tool_name == "check_assembly":
                return check_assembly_tool(
                    build_dir=cmake_actions.build_dir,
                    project_path=cmake_actions.project_path,
                    unavoidable_asm_path=cmake_actions.unavoidable_asm_path,
                    verbose=self.verbose,
                )
            # Install packages
            elif tool_name == "install_packages":
                return install_packages(
                    current_image=cmake_actions.container_image,
                    packages=tool_input.get("packages", []),
                    verbose=self.verbose,
                )
            # Finish tool
            elif tool_name == "finish":
                return {
                    "acknowledged": True,
                    "status": tool_input.get("status"),
                    "summary": tool_input.get("summary"),
                }
            # Request more turns (handled by state machine in _execute_react_loop)
            elif tool_name == "request_more_turns":
                return {"acknowledged": True}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except ValueError as e:
            return {"error": f"Security error: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool error: {str(e)}"}

    def _get_default_cmake_config(self) -> list[str]:
        """Get default CMake configuration (fallback if LLM fails)."""
        flags = [
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_C_COMPILER=clang-18",
            "-DCMAKE_CXX_COMPILER=clang++-18",
        ]

        if self.enable_lto:
            if self.generate_ir:
                flags.extend([
                    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
                    "-DCMAKE_C_FLAGS=-g -flto=full -save-temps=obj",
                    "-DCMAKE_CXX_FLAGS=-g -flto=full -save-temps=obj",
                ])
            else:
                flags.extend([
                    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
                    "-DCMAKE_C_FLAGS=-g -flto=full",
                    "-DCMAKE_CXX_FLAGS=-g -flto=full",
                ])

        if self.prefer_static:
            flags.append("-DBUILD_SHARED_LIBS=OFF")

        return flags

    def _attempt_cmake_build(self, project_path: Path, cmake_flags: list[str]) -> str:
        """
        Attempt to build the project with given CMake flags.

        Uses two-phase approach: configure then build.
        Raises BuildError if either phase fails.
        Returns combined log on success.
        """
        if self.build_dir:
            build_dir = Path(self.build_dir).resolve()
        else:
            build_dir = project_path / "build"

        unavoidable_asm_path = self._get_unavoidable_asm_path(project_path.name)
        actions = CMakeActions(
            project_path, build_dir, self.container_image,
            unavoidable_asm_path=unavoidable_asm_path, verbose=self.verbose,
            ccache_dir=self.ccache_dir,
        )

        if self.verbose:
            print(f"[Docker] Build directory: {build_dir}")
            print("[Phase 1/2] Running CMake configure...")

        configure_result = actions.cmake_configure(cmake_flags)

        if not configure_result["success"]:
            error_msg = configure_result.get("error", "Unknown configure error")
            output = configure_result.get("output", "")
            raise BuildError(f"{error_msg}\n{output}")

        configure_output = configure_result["output"]

        if self.verbose:
            print("[Phase 2/2] Running ninja build...")

        build_result = actions.cmake_build()

        if not build_result["success"]:
            error_msg = build_result.get("error", "Unknown build error")
            output = build_result.get("output", "")
            raise BuildError(f"{error_msg}\n{output}")

        build_output = build_result["output"]

        return (
            f"=== Configure Output ===\n{configure_output}"
            f"\n\n=== Build Output ===\n{build_output}"
        )

    def _apply_suggestions(
        self, current_flags: list[str], analysis: dict
    ) -> list[str]:
        """Apply LLM suggestions to the current configuration."""
        flags = current_flags.copy()

        for suggested_flag in analysis.get("suggested_flags", []):
            if suggested_flag.startswith("-D"):
                key = suggested_flag.split("=")[0]
                flags = [f for f in flags if not f.startswith(key)]
            flags.append(suggested_flag)

        for cmake_var, new_value in analysis.get("compiler_flag_changes", {}).items():
            flags = [f for f in flags if not f.startswith(cmake_var)]
            flags.append(f"{cmake_var}={new_value}")

        missing_deps = analysis.get("missing_dependencies", [])
        if missing_deps and self.verbose:
            print("[Note] The following dependencies are missing from the Docker image:")
            for dep in missing_deps:
                print(f"  - {dep}")
            print(
                "[Note] Update the Dockerfile to include "
                "these dependencies and rebuild the image"
            )

        return flags

    def extract_compile_commands(
        self, project_path: Path, output_dir: Path | None = None, use_build_dir: bool = True
    ) -> Path:
        """
        Extract compile_commands.json from build directory to output directory.

        Fixes Docker container paths to host paths and preserves generated
        source files (e.g. protobuf .pb.cc, lex/yacc output) that only exist
        in the build directory by copying them into output_dir/sources/.

        Files that already exist in the project source tree are remapped
        without copying.  Entries whose source file cannot be found anywhere
        are dropped.

        Args:
            project_path: Path to the project source
            output_dir: Directory to save compile_commands.json (default: build-scripts/<project>/)
            use_build_dir: Whether out-of-source build was used

        Returns the path to the extracted file.
        """
        if use_build_dir:
            if self.build_dir:
                build_dir = Path(self.build_dir).resolve()
            else:
                build_dir = project_path / "build"
            compile_commands_src = build_dir / "compile_commands.json"
        else:
            build_dir = project_path
            compile_commands_src = project_path / "compile_commands.json"

        if output_dir is None:
            output_dir = Path("build-scripts") / project_path.name
            output_dir.mkdir(parents=True, exist_ok=True)

        compile_commands_dst = output_dir / "compile_commands.json"

        if not compile_commands_src.exists():
            raise FileNotFoundError(
                f"compile_commands.json not found in {compile_commands_src.parent}"
            )

        with open(compile_commands_src) as f:
            compile_commands = json.load(f)

        pp = str(project_path)
        bd = str(build_dir)

        def _replace(s: str) -> str:
            return s.replace(DOCKER_WORKSPACE_SRC, pp).replace(DOCKER_WORKSPACE_BUILD, bd)

        def _replace_file_path(entry: dict, old_path: str, new_path: str) -> None:
            """Replace a file path in all fields of an entry."""
            entry["file"] = new_path
            if "command" in entry:
                entry["command"] = entry["command"].replace(old_path, new_path)
            if "arguments" in entry:
                entry["arguments"] = [a.replace(old_path, new_path) for a in entry["arguments"]]

        sources_dir = output_dir / "sources"
        fixed = []
        removed = 0
        copied = 0

        for entry in compile_commands:
            # -- Phase 1: Replace Docker container paths in all fields --
            for field in ("file", "directory", "output"):
                if field in entry:
                    entry[field] = _replace(entry[field])
            if "command" in entry:
                entry["command"] = _replace(entry["command"])
            if "arguments" in entry:
                entry["arguments"] = [_replace(a) for a in entry["arguments"]]

            # -- Phase 2: Resolve relative file paths through Docker-space --
            file_path = entry.get("file", "")
            directory = entry.get("directory", "")
            if file_path and not Path(file_path).is_absolute() and directory:
                docker_dir = directory.replace(bd, DOCKER_WORKSPACE_BUILD).replace(
                    pp, DOCKER_WORKSPACE_SRC
                )
                docker_resolved = str((Path(docker_dir) / file_path).resolve())
                resolved = _replace(docker_resolved)
                _replace_file_path(entry, file_path, resolved)
                file_path = resolved

            # -- Phase 3: Remap build-dir aliases to source tree --
            # Handles cases like ffmpeg where a src/ symlink in the build dir
            # points back to the source tree (broken on host).
            if file_path and not Path(file_path).exists() and file_path.startswith(bd + "/"):
                rel = file_path[len(bd) + 1:]
                for candidate_rel in [rel] + [
                    rel[len(p):] for p in ("src/", "source/") if rel.startswith(p)
                ]:
                    candidate = str(project_path / candidate_rel)
                    if Path(candidate).exists():
                        _replace_file_path(entry, file_path, candidate)
                        file_path = candidate
                        break

            # -- Phase 4: Copy build-only generated files --
            # If the file doesn't exist on host but does in the build dir,
            # copy it to output_dir/sources/ to preserve it.
            if file_path and not Path(file_path).exists():
                # Translate the host path back to the build dir to find the
                # actual file (it may have been remapped to project_path in
                # phase 1 but only exists in build_dir).
                build_candidates = [file_path]
                if file_path.startswith(pp + "/"):
                    rel = file_path[len(pp) + 1:]
                    build_candidates.append(str(build_dir / rel))
                # The original Docker path may also help
                for bc in build_candidates:
                    if Path(bc).exists():
                        rel = (
                            Path(bc).relative_to(build_dir)
                            if bc.startswith(bd)
                            else Path(bc).name
                        )
                        dest = sources_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(bc, dest)
                        _replace_file_path(entry, file_path, str(dest))
                        file_path = str(dest)
                        copied += 1
                        break

            # -- Phase 5: Drop entries whose source file can't be found --
            if file_path and not Path(file_path).exists():
                removed += 1
                continue

            fixed.append(entry)

        with open(compile_commands_dst, "w") as f:
            json.dump(fixed, f, indent=2)

        if self.verbose:
            print(f"[Extract] Saved {len(fixed)} entries to {compile_commands_dst}")
            if copied:
                print(f"[Extract] Copied {copied} generated source files to {sources_dir}")
            if removed:
                print(f"[Extract] Removed {removed} entries with missing source files")

        return compile_commands_dst
