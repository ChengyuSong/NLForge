"""Standard library allocation, free, initialization, and safety summaries."""

from .models import (
    Allocation,
    AllocationSummary,
    AllocationType,
    FreeOp,
    FreeSummary,
    InitOp,
    InitSummary,
    MemsafeContract,
    MemsafeSummary,
    ParameterInfo,
)

# Pre-defined summaries for common C standard library functions
STDLIB_SUMMARIES: dict[str, AllocationSummary] = {
    # Memory allocation
    "malloc": AllocationSummary(
        function_name="malloc",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="malloc",
                size_expr="size",
                size_params=["size"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={"size": ParameterInfo(role="size_indicator", used_in_allocation=True)},
        description="Allocates size bytes of uninitialized heap memory.",
    ),
    "calloc": AllocationSummary(
        function_name="calloc",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="calloc",
                size_expr="nmemb * size",
                size_params=["nmemb", "size"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "nmemb": ParameterInfo(role="count", used_in_allocation=True),
            "size": ParameterInfo(role="element_size", used_in_allocation=True),
        },
        description="Allocates zero-initialized memory for nmemb elements of size bytes each.",
    ),
    "realloc": AllocationSummary(
        function_name="realloc",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="realloc",
                size_expr="size",
                size_params=["size"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "ptr": ParameterInfo(role="existing_buffer", used_in_allocation=False),
            "size": ParameterInfo(role="new_size", used_in_allocation=True),
        },
        description="Changes the size of the memory block pointed to by ptr to size bytes.",
    ),
    "reallocarray": AllocationSummary(
        function_name="reallocarray",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="reallocarray",
                size_expr="nmemb * size",
                size_params=["nmemb", "size"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "ptr": ParameterInfo(role="existing_buffer", used_in_allocation=False),
            "nmemb": ParameterInfo(role="count", used_in_allocation=True),
            "size": ParameterInfo(role="element_size", used_in_allocation=True),
        },
        description="Reallocates memory for an array with overflow checking.",
    ),
    "free": AllocationSummary(
        function_name="free",
        allocations=[],
        parameters={"ptr": ParameterInfo(role="pointer_to_free", used_in_allocation=False)},
        description="Frees memory previously allocated by malloc/calloc/realloc.",
    ),
    "aligned_alloc": AllocationSummary(
        function_name="aligned_alloc",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="aligned_alloc",
                size_expr="size",
                size_params=["size"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "alignment": ParameterInfo(role="alignment", used_in_allocation=False),
            "size": ParameterInfo(role="size_indicator", used_in_allocation=True),
        },
        description="Allocates size bytes aligned to alignment boundary.",
    ),
    # String functions that allocate
    "strdup": AllocationSummary(
        function_name="strdup",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="strdup",
                size_expr="strlen(s) + 1",
                size_params=["s"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={"s": ParameterInfo(role="source_string", used_in_allocation=True)},
        description="Duplicates string s, allocating memory for the copy.",
    ),
    "strndup": AllocationSummary(
        function_name="strndup",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="strndup",
                size_expr="min(strlen(s), n) + 1",
                size_params=["s", "n"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "s": ParameterInfo(role="source_string", used_in_allocation=True),
            "n": ParameterInfo(role="max_length", used_in_allocation=True),
        },
        description="Duplicates at most n bytes of string s.",
    ),
    "asprintf": AllocationSummary(
        function_name="asprintf",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="asprintf",
                size_expr="formatted string length + 1",
                size_params=["format"],
                returned=False,
                stored_to="*strp",
                may_be_null=True,
            )
        ],
        parameters={
            "strp": ParameterInfo(role="output_pointer", used_in_allocation=False),
            "format": ParameterInfo(role="format_string", used_in_allocation=True),
        },
        description="Allocates and prints formatted string to *strp.",
    ),
    # POSIX functions
    "getline": AllocationSummary(
        function_name="getline",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="getline",
                size_expr="line length",
                size_params=[],
                returned=False,
                stored_to="*lineptr",
                may_be_null=True,
            )
        ],
        parameters={
            "lineptr": ParameterInfo(role="output_buffer", used_in_allocation=False),
            "n": ParameterInfo(role="buffer_size", used_in_allocation=True),
            "stream": ParameterInfo(role="input_stream", used_in_allocation=False),
        },
        description="Reads a line, allocating/reallocating buffer as needed.",
    ),
    # File operations (for completeness)
    "fopen": AllocationSummary(
        function_name="fopen",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="fopen",
                size_expr="sizeof(FILE)",
                size_params=[],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "pathname": ParameterInfo(role="file_path", used_in_allocation=False),
            "mode": ParameterInfo(role="open_mode", used_in_allocation=False),
        },
        description="Opens file and returns FILE stream (allocated internally).",
    ),
    "fdopen": AllocationSummary(
        function_name="fdopen",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="fdopen",
                size_expr="sizeof(FILE)",
                size_params=[],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "fd": ParameterInfo(role="file_descriptor", used_in_allocation=False),
            "mode": ParameterInfo(role="open_mode", used_in_allocation=False),
        },
        description="Associates FILE stream with file descriptor.",
    ),
    "tmpfile": AllocationSummary(
        function_name="tmpfile",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="tmpfile",
                size_expr="sizeof(FILE)",
                size_params=[],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={},
        description="Creates and opens a temporary file.",
    ),
    "fclose": AllocationSummary(
        function_name="fclose",
        allocations=[],
        parameters={"stream": ParameterInfo(role="file_stream", used_in_allocation=False)},
        description="Closes file stream and frees associated resources.",
    ),
    # Directory operations
    "opendir": AllocationSummary(
        function_name="opendir",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="opendir",
                size_expr="sizeof(DIR)",
                size_params=[],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={"name": ParameterInfo(role="directory_path", used_in_allocation=False)},
        description="Opens directory stream.",
    ),
    "closedir": AllocationSummary(
        function_name="closedir",
        allocations=[],
        parameters={"dirp": ParameterInfo(role="directory_stream", used_in_allocation=False)},
        description="Closes directory stream.",
    ),
    # Memory mapping
    "mmap": AllocationSummary(
        function_name="mmap",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="mmap",
                size_expr="length",
                size_params=["length"],
                returned=True,
                may_be_null=True,  # Returns MAP_FAILED on error
            )
        ],
        parameters={
            "addr": ParameterInfo(role="hint_address", used_in_allocation=False),
            "length": ParameterInfo(role="size_indicator", used_in_allocation=True),
            "prot": ParameterInfo(role="protection", used_in_allocation=False),
            "flags": ParameterInfo(role="mapping_flags", used_in_allocation=False),
            "fd": ParameterInfo(role="file_descriptor", used_in_allocation=False),
            "offset": ParameterInfo(role="file_offset", used_in_allocation=False),
        },
        description="Maps memory region of length bytes.",
    ),
    "munmap": AllocationSummary(
        function_name="munmap",
        allocations=[],
        parameters={
            "addr": ParameterInfo(role="mapped_address", used_in_allocation=False),
            "length": ParameterInfo(role="mapping_length", used_in_allocation=False),
        },
        description="Unmaps memory region.",
    ),
}


# Pre-defined free summaries for common C standard library functions
STDLIB_FREE_SUMMARIES: dict[str, FreeSummary] = {
    "free": FreeSummary(
        function_name="free",
        frees=[
            FreeOp(
                target="ptr",
                target_kind="parameter",
                deallocator="free",
                conditional=False,
                nulled_after=False,
            )
        ],
        description="Frees heap memory previously allocated by malloc/calloc/realloc.",
    ),
    "realloc": FreeSummary(
        function_name="realloc",
        frees=[
            FreeOp(
                target="ptr",
                target_kind="parameter",
                deallocator="realloc",
                conditional=False,
                nulled_after=False,
            )
        ],
        description="Frees old pointer when reallocating to a new size.",
    ),
    "fclose": FreeSummary(
        function_name="fclose",
        frees=[
            FreeOp(
                target="stream",
                target_kind="parameter",
                deallocator="fclose",
                conditional=False,
                nulled_after=False,
            )
        ],
        description="Closes file stream and frees associated FILE structure.",
    ),
    "closedir": FreeSummary(
        function_name="closedir",
        frees=[
            FreeOp(
                target="dirp",
                target_kind="parameter",
                deallocator="closedir",
                conditional=False,
                nulled_after=False,
            )
        ],
        description="Closes directory stream and frees associated DIR structure.",
    ),
    "munmap": FreeSummary(
        function_name="munmap",
        frees=[
            FreeOp(
                target="addr",
                target_kind="parameter",
                deallocator="munmap",
                conditional=False,
                nulled_after=False,
            )
        ],
        description="Unmaps memory region previously mapped with mmap.",
    ),
    "freeaddrinfo": FreeSummary(
        function_name="freeaddrinfo",
        frees=[
            FreeOp(
                target="res",
                target_kind="parameter",
                deallocator="freeaddrinfo",
                conditional=False,
                nulled_after=False,
            )
        ],
        description="Frees addrinfo linked list returned by getaddrinfo.",
    ),
}


# Pre-defined init summaries for common C standard library functions
STDLIB_INIT_SUMMARIES: dict[str, InitSummary] = {
    "calloc": InitSummary(
        function_name="calloc",
        inits=[
            InitOp(
                target="return value",
                target_kind="return_value",
                initializer="calloc",
                byte_count="nmemb * size",
            )
        ],
        description="Always initializes returned buffer to zero.",
    ),
    "memset": InitSummary(
        function_name="memset",
        inits=[
            InitOp(
                target="*s",
                target_kind="parameter",
                initializer="memset",
                byte_count="n",
            )
        ],
        description="Always initializes s with byte value c for n bytes.",
    ),
    "memcpy": InitSummary(
        function_name="memcpy",
        inits=[
            InitOp(
                target="*dest",
                target_kind="parameter",
                initializer="memcpy",
                byte_count="n",
            )
        ],
        description="Always initializes dest by copying n bytes from src.",
    ),
    "memmove": InitSummary(
        function_name="memmove",
        inits=[
            InitOp(
                target="*dest",
                target_kind="parameter",
                initializer="memmove",
                byte_count="n",
            )
        ],
        description="Always initializes dest by copying n bytes from src (overlapping safe).",
    ),
    "strncpy": InitSummary(
        function_name="strncpy",
        inits=[
            InitOp(
                target="*dest",
                target_kind="parameter",
                initializer="strncpy",
                byte_count="n",
            )
        ],
        description="Always initializes dest with up to n bytes from src, zero-padded.",
    ),
    "snprintf": InitSummary(
        function_name="snprintf",
        inits=[
            InitOp(
                target="*str",
                target_kind="parameter",
                initializer="snprintf",
                byte_count="up to size",
            )
        ],
        description="Always initializes str with formatted output, null-terminated.",
    ),
    "strdup": InitSummary(
        function_name="strdup",
        inits=[
            InitOp(
                target="return value",
                target_kind="return_value",
                initializer="strdup",
                byte_count="strlen(s) + 1",
            )
        ],
        description="Always initializes returned buffer with copy of input string.",
    ),
    "strndup": InitSummary(
        function_name="strndup",
        inits=[
            InitOp(
                target="return value",
                target_kind="return_value",
                initializer="strndup",
                byte_count="min(n, strlen(s)) + 1",
            )
        ],
        description=(
            "Always initializes returned buffer with copy of"
            " up to n bytes of input string."
        ),
    ),
}


# Pre-defined safety contract summaries for common C standard library functions
STDLIB_MEMSAFE_SUMMARIES: dict[str, MemsafeSummary] = {
    "memcpy": MemsafeSummary(
        function_name="memcpy",
        contracts=[
            MemsafeContract(
                target="dest", contract_kind="not_null",
                description="dest must not be NULL",
            ),
            MemsafeContract(
                target="dest", contract_kind="buffer_size",
                description="dest must point to at least n bytes",
                size_expr="n", relationship="byte_count",
            ),
            MemsafeContract(
                target="src", contract_kind="not_null",
                description="src must not be NULL",
            ),
            MemsafeContract(
                target="src", contract_kind="buffer_size",
                description="src must point to at least n bytes",
                size_expr="n", relationship="byte_count",
            ),
        ],
        description=(
            "Requires both dest and src to be non-NULL"
            " and point to at least n bytes."
        ),
    ),
    "memmove": MemsafeSummary(
        function_name="memmove",
        contracts=[
            MemsafeContract(
                target="dest", contract_kind="not_null",
                description="dest must not be NULL",
            ),
            MemsafeContract(
                target="dest", contract_kind="buffer_size",
                description="dest must point to at least n bytes",
                size_expr="n", relationship="byte_count",
            ),
            MemsafeContract(
                target="src", contract_kind="not_null",
                description="src must not be NULL",
            ),
            MemsafeContract(
                target="src", contract_kind="buffer_size",
                description="src must point to at least n bytes",
                size_expr="n", relationship="byte_count",
            ),
        ],
        description=(
            "Requires both dest and src to be non-NULL"
            " and point to at least n bytes."
        ),
    ),
    "memset": MemsafeSummary(
        function_name="memset",
        contracts=[
            MemsafeContract(
                target="s", contract_kind="not_null",
                description="s must not be NULL",
            ),
            MemsafeContract(
                target="s", contract_kind="buffer_size",
                description="s must point to at least n bytes",
                size_expr="n", relationship="byte_count",
            ),
        ],
        description="Requires s to be non-NULL and point to at least n bytes.",
    ),
    "free": MemsafeSummary(
        function_name="free",
        contracts=[
            MemsafeContract(
                target="ptr", contract_kind="nullable",
                description=(
                    "ptr may be NULL"
                    " (free(NULL) is a no-op per C standard)"
                ),
            ),
            MemsafeContract(
                target="ptr", contract_kind="not_freed",
                description=(
                    "if non-NULL, ptr must point to live"
                    " heap memory (not already freed)"
                ),
            ),
        ],
        description=(
            "Accepts NULL (no-op). If non-NULL,"
            " ptr must point to live heap memory."
        ),
    ),
    "strlen": MemsafeSummary(
        function_name="strlen",
        contracts=[
            MemsafeContract(
                target="s", contract_kind="not_null",
                description="s must not be NULL",
            ),
            MemsafeContract(
                target="s", contract_kind="initialized",
                description=(
                    "s must point to an initialized,"
                    " null-terminated string"
                ),
            ),
        ],
        description=(
            "Requires s to be a non-NULL, initialized,"
            " null-terminated string."
        ),
    ),
    "strcpy": MemsafeSummary(
        function_name="strcpy",
        contracts=[
            MemsafeContract(
                target="dest", contract_kind="not_null",
                description="dest must not be NULL",
            ),
            MemsafeContract(
                target="dest", contract_kind="buffer_size",
                description=(
                    "dest must have space for"
                    " strlen(src)+1 bytes"
                ),
                size_expr="strlen(src)+1",
                relationship="byte_count",
            ),
            MemsafeContract(
                target="src", contract_kind="not_null",
                description="src must not be NULL",
            ),
            MemsafeContract(
                target="src", contract_kind="initialized",
                description=(
                    "src must be an initialized,"
                    " null-terminated string"
                ),
            ),
        ],
        description=(
            "Requires dest to have sufficient space"
            " and src to be a valid string."
        ),
    ),
    "strncpy": MemsafeSummary(
        function_name="strncpy",
        contracts=[
            MemsafeContract(
                target="dest", contract_kind="not_null",
                description="dest must not be NULL",
            ),
            MemsafeContract(
                target="dest", contract_kind="buffer_size",
                description="dest must point to at least n bytes",
                size_expr="n", relationship="byte_count",
            ),
            MemsafeContract(
                target="src", contract_kind="not_null",
                description="src must not be NULL",
            ),
        ],
        description=(
            "Requires dest to have at least n bytes"
            " and src to be non-NULL."
        ),
    ),
    "strcmp": MemsafeSummary(
        function_name="strcmp",
        contracts=[
            MemsafeContract(
                target="s1", contract_kind="not_null",
                description="s1 must not be NULL",
            ),
            MemsafeContract(
                target="s1", contract_kind="initialized",
                description=(
                    "s1 must be an initialized,"
                    " null-terminated string"
                ),
            ),
            MemsafeContract(
                target="s2", contract_kind="not_null",
                description="s2 must not be NULL",
            ),
            MemsafeContract(
                target="s2", contract_kind="initialized",
                description=(
                    "s2 must be an initialized,"
                    " null-terminated string"
                ),
            ),
        ],
        description=(
            "Requires both s1 and s2 to be"
            " non-NULL, initialized strings."
        ),
    ),
    "snprintf": MemsafeSummary(
        function_name="snprintf",
        contracts=[
            MemsafeContract(
                target="str", contract_kind="not_null",
                description="str must not be NULL",
            ),
            MemsafeContract(
                target="str", contract_kind="buffer_size",
                description=(
                    "str must point to at least size bytes"
                ),
                size_expr="size", relationship="byte_count",
            ),
            MemsafeContract(
                target="format", contract_kind="not_null",
                description="format must not be NULL",
            ),
        ],
        description=(
            "Writes at most size bytes to str (including the null terminator);"
            " output is truncated if the formatted result would exceed size-1"
            " chars — str can never overflow regardless of argument values."
            " Requires str to point to at least size bytes and format to be non-NULL."
        ),
    ),
    "printf": MemsafeSummary(
        function_name="printf",
        contracts=[
            MemsafeContract(
                target="format", contract_kind="not_null",
                description="format must not be NULL",
            ),
        ],
        description="Requires format to be non-NULL.",
    ),
    "fprintf": MemsafeSummary(
        function_name="fprintf",
        contracts=[
            MemsafeContract(
                target="stream", contract_kind="not_null",
                description="stream must not be NULL",
            ),
            MemsafeContract(
                target="format", contract_kind="not_null",
                description="format must not be NULL",
            ),
        ],
        description=(
            "Requires stream and format to be non-NULL."
        ),
    ),
    "fwrite": MemsafeSummary(
        function_name="fwrite",
        contracts=[
            MemsafeContract(
                target="ptr", contract_kind="not_null",
                description="ptr must not be NULL",
            ),
            MemsafeContract(
                target="ptr", contract_kind="buffer_size",
                description=(
                    "ptr must point to at least"
                    " size*nmemb bytes"
                ),
                size_expr="size*nmemb",
                relationship="byte_count",
            ),
            MemsafeContract(
                target="stream", contract_kind="not_null",
                description="stream must not be NULL",
            ),
        ],
        description=(
            "Requires ptr to have at least size*nmemb"
            " bytes and stream to be non-NULL."
        ),
    ),
    "fread": MemsafeSummary(
        function_name="fread",
        contracts=[
            MemsafeContract(
                target="ptr", contract_kind="not_null",
                description="ptr must not be NULL",
            ),
            MemsafeContract(
                target="ptr", contract_kind="buffer_size",
                description=(
                    "ptr must point to at least"
                    " size*nmemb bytes"
                ),
                size_expr="size*nmemb",
                relationship="byte_count",
            ),
            MemsafeContract(
                target="stream", contract_kind="not_null",
                description="stream must not be NULL",
            ),
        ],
        description=(
            "Requires ptr to have at least size*nmemb"
            " bytes and stream to be non-NULL."
        ),
    ),
    "malloc": MemsafeSummary(
        function_name="malloc",
        contracts=[],
        description="No safety pre-conditions required (size=0 is valid).",
    ),
}


# Function attributes for stdlib functions
STDLIB_ATTRIBUTES: dict[str, str] = {
    # noreturn functions
    "exit": "__attribute__((noreturn))",
    "abort": "__attribute__((noreturn))",
    "_exit": "__attribute__((noreturn))",
    "_Exit": "__attribute__((noreturn))",
    "quick_exit": "__attribute__((noreturn))",
    "__assert_fail": "__attribute__((noreturn))",
    "pthread_exit": "__attribute__((noreturn))",
    "longjmp": "__attribute__((noreturn))",
    "siglongjmp": "__attribute__((noreturn))",
    "err": "__attribute__((noreturn))",
    "errx": "__attribute__((noreturn))",
    "verr": "__attribute__((noreturn))",
    "verrx": "__attribute__((noreturn))",
    "thrd_exit": "__attribute__((noreturn))",
    # nonnull + returns_nonnull
    "strlen": "__attribute__((nonnull(1)))",
    "strcmp": "__attribute__((nonnull(1, 2)))",
    "strncmp": "__attribute__((nonnull(1, 2)))",
    "strcpy": "__attribute__((nonnull(1, 2)))",
    "strncpy": "__attribute__((nonnull(1, 2)))",
    "memcpy": "__attribute__((nonnull(1, 2), returns_nonnull))",
    "memmove": "__attribute__((nonnull(1, 2), returns_nonnull))",
    "memset": "__attribute__((nonnull(1), returns_nonnull))",
}


def get_stdlib_attributes(name: str) -> str:
    """Get attribute text for a stdlib function, or empty string."""
    return STDLIB_ATTRIBUTES.get(name, "")


def get_all_stdlib_attributes() -> dict[str, str]:
    """Get all stdlib function attributes."""
    return STDLIB_ATTRIBUTES.copy()


def get_stdlib_memsafe_summary(name: str) -> MemsafeSummary | None:
    """Get pre-defined safety summary for a standard library function."""
    return STDLIB_MEMSAFE_SUMMARIES.get(name)


def get_all_stdlib_memsafe_summaries() -> dict[str, MemsafeSummary]:
    """Get all pre-defined stdlib safety summaries."""
    return STDLIB_MEMSAFE_SUMMARIES.copy()


def get_stdlib_init_summary(name: str) -> InitSummary | None:
    """Get pre-defined init summary for a standard library function."""
    return STDLIB_INIT_SUMMARIES.get(name)


def get_all_stdlib_init_summaries() -> dict[str, InitSummary]:
    """Get all pre-defined stdlib init summaries."""
    return STDLIB_INIT_SUMMARIES.copy()


def get_stdlib_free_summary(name: str) -> FreeSummary | None:
    """Get pre-defined free summary for a standard library function."""
    return STDLIB_FREE_SUMMARIES.get(name)


def get_all_stdlib_free_summaries() -> dict[str, FreeSummary]:
    """Get all pre-defined stdlib free summaries."""
    return STDLIB_FREE_SUMMARIES.copy()


def get_stdlib_summary(name: str) -> AllocationSummary | None:
    """Get pre-defined summary for a standard library function."""
    return STDLIB_SUMMARIES.get(name)


def get_all_stdlib_summaries() -> dict[str, AllocationSummary]:
    """Get all pre-defined stdlib summaries."""
    return STDLIB_SUMMARIES.copy()


def is_stdlib_allocator(name: str) -> bool:
    """Check if a function name is a known stdlib allocator."""
    summary = STDLIB_SUMMARIES.get(name)
    if summary is None:
        return False
    return len(summary.allocations) > 0


class StdlibSummaryGenerator:
    """
    Generates stdlib summaries from documentation using LLM.

    This can be used to expand the stdlib coverage by parsing
    man pages or documentation.
    """

    def __init__(self, llm):
        from .llm.base import LLMBackend

        self.llm: LLMBackend = llm

    def generate_from_manpage(self, manpage_content: str) -> list[AllocationSummary]:
        """
        Generate summaries from man page content.

        Args:
            manpage_content: Text content of a man page

        Returns:
            List of generated summaries
        """
        prompt = f"""Analyze this man page and identify functions that allocate memory.
For each allocating function, provide a summary in JSON format.

Man page content:
```
{manpage_content}
```

For each function that allocates memory, provide:
```json
{{
  "function": "function_name",
  "allocations": [
    {{
      "type": "heap",
      "source": "function_name",
      "size_expr": "expression for size",
      "size_params": ["parameters affecting size"],
      "returned": true,
      "stored_to": null,
      "may_be_null": true
    }}
  ],
  "parameters": {{
    "param": {{"role": "description", "used_in_allocation": true}}
  }},
  "description": "One-sentence description"
}}
```

Return a JSON array of summaries, or an empty array if no functions allocate memory.
"""
        response = self.llm.complete(prompt)
        return self._parse_summaries(response)

    def _parse_summaries(self, response: str) -> list[AllocationSummary]:
        """Parse LLM response into summaries."""
        import json
        import re

        # Find JSON array in response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []

        summaries = []
        for item in data:
            allocations = []
            for a in item.get("allocations", []):
                try:
                    alloc_type = AllocationType(a.get("type", "unknown"))
                except ValueError:
                    alloc_type = AllocationType.UNKNOWN

                allocations.append(
                    Allocation(
                        alloc_type=alloc_type,
                        source=a.get("source", ""),
                        size_expr=a.get("size_expr"),
                        size_params=a.get("size_params", []),
                        returned=a.get("returned", False),
                        stored_to=a.get("stored_to"),
                        may_be_null=a.get("may_be_null", True),
                    )
                )

            parameters = {}
            for name, info in item.get("parameters", {}).items():
                parameters[name] = ParameterInfo(
                    role=info.get("role", ""),
                    used_in_allocation=info.get("used_in_allocation", False),
                )

            summaries.append(
                AllocationSummary(
                    function_name=item.get("function", ""),
                    allocations=allocations,
                    parameters=parameters,
                    description=item.get("description", ""),
                )
            )

        return summaries
