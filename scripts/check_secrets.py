from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data",
    "models",
}

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}['\"]", re.IGNORECASE),
]

TEXT_SUFFIXES = {
    ".env",
    ".example",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def main() -> None:
    findings: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or _should_skip(path):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{_line_no(text, match.start())}")

    if findings:
        print("Potential secrets found:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("Secret scan passed: no obvious API keys or tokens found.")


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(PROJECT_ROOT).parts[:-1])


def _line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


if __name__ == "__main__":
    main()
