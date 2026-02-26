"""Tool schemas for the link-unit discovery agent (Anthropic tool-use format)."""

FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file. Use 'build/' prefix for build artifacts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path (e.g. 'build/Makefile', 'configure.ac')",
                },
                "max_lines": {"type": "integer", "default": 200},
                "start_line": {"type": "integer", "default": 1},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files in a directory. Use 'build/' prefix for build dir.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "Relative path (e.g. '.', 'build', 'build/bfd')",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '*.a', 'Makefile*')",
                },
            },
        },
    },
]

SKILL_TOOLS = [
    {
        "name": "parse_ninja_targets",
        "description": "Parse build.ninja to extract all link targets with objects and deps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "build_ninja_path": {
                    "type": "string",
                    "description": "Path to build.ninja (e.g. 'build/build.ninja')",
                },
            },
            "required": ["build_ninja_path"],
        },
    },
    {
        "name": "run_ar_t",
        "description": "List member object files in a static archive (.a file).",
        "input_schema": {
            "type": "object",
            "properties": {
                "archive_path": {
                    "type": "string",
                    "description": "Path to .a archive (use 'build/' prefix)",
                },
            },
            "required": ["archive_path"],
        },
    },
    {
        "name": "find_files",
        "description": "Find files matching a glob pattern recursively.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.a', '**/Makefile')",
                },
            },
            "required": ["directory", "pattern"],
        },
    },
]

COMMAND_TOOL = {
    "name": "run_command",
    "description": (
        "Run a shell command in a Docker container with read-only access to "
        "/workspace/src and /workspace/build. Use for grep, find, make -n, nm, "
        "readelf, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command (via bash -c)",
            },
            "workdir": {
                "type": "string",
                "enum": ["build", "src"],
                "default": "build",
            },
        },
        "required": ["command"],
    },
}

REQUEST_MORE_TURNS_TOOL = {
    "name": "request_more_turns",
    "description": (
        "Request additional turns when running low but still making progress. "
        "Use when you have identified some link units but need more turns to resolve "
        "remaining targets. Grants 10 extra turns per call (max 10 extensions)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why more turns are needed and what remains to be done.",
            },
        },
        "required": ["reason"],
    },
}

FINISH_TOOL = {
    "name": "finish",
    "description": "Call when done. Provide discovered link units. No .bc files needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "partial", "failure"],
            },
            "link_units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["static_library", "shared_library", "executable"],
                        },
                        "output": {"type": "string", "description": "Output path relative to build dir"},
                        "objects": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "link_deps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Project library dependencies",
                        },
                    },
                    "required": ["name", "type", "output"],
                },
            },
            "build_system": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["status", "link_units"],
    },
}

DISCOVERY_TOOL_DEFINITIONS = FILE_TOOLS + SKILL_TOOLS + [COMMAND_TOOL, REQUEST_MORE_TURNS_TOOL, FINISH_TOOL]
