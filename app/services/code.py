"""Helpers for the 'code' source type.

Maps file extensions to language names so chunk metadata can carry
{"language": "python"} alongside file_path/start_line/end_line.

Intentionally simple — a hand-maintained dict, not a full
language-detection library. Tree-sitter / symbol-aware chunking
is a Phase 2 follow-up.
"""

_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-c",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".rst": "rst",
    ".tex": "latex",
    ".r": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".hs": "haskell",
    ".elm": "elm",
    ".dart": "dart",
    ".zig": "zig",
    ".nim": "nim",
    ".v": "v",
    ".sol": "solidity",
    ".dockerfile": "dockerfile",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
}

_FILENAME_TO_LANGUAGE = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "rakefile": "ruby",
    "gemfile": "ruby",
}


def language_for_path(file_path: str) -> str:
    """Return a language name for a file path, falling back to 'plaintext'.

    Looks at the extension first, then the basename for ext-less files
    like Dockerfile or Makefile.
    """
    from pathlib import Path

    p = Path(file_path)
    ext = p.suffix.lower()
    if ext in _EXT_TO_LANGUAGE:
        return _EXT_TO_LANGUAGE[ext]

    name = p.name.lower()
    if name in _FILENAME_TO_LANGUAGE:
        return _FILENAME_TO_LANGUAGE[name]

    return "plaintext"
