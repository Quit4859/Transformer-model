"""Small, permission-aware workspace tools used by the agent harness."""

from __future__ import annotations

import difflib
import glob as glob_module
import shlex
import subprocess
from pathlib import Path
from typing import Any


class PermissionError(RuntimeError):
    pass


class WorkspaceTools:
    def __init__(self, root: str | Path, *, allow_write: bool = False, allow_bash: bool = False):
        self.root = Path(root).resolve()
        self.allow_write = allow_write
        self.allow_bash = allow_bash

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("path escapes workspace")
        return path

    def read(self, path: str) -> str:
        return self._path(path).read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> None:
        if not self.allow_write:
            raise PermissionError("write permission is disabled")
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def edit(self, path: str, old: str, new: str) -> None:
        content = self.read(path)
        if old not in content:
            raise ValueError("edit target was not found")
        self.write(path, content.replace(old, new, 1))

    def diff(self, path: str, old: str, new: str) -> str:
        return "".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), fromfile=path, tofile=path
        ))

    def glob(self, pattern: str) -> list[str]:
        return sorted(
            str(Path(path).resolve().relative_to(self.root))
            for path in glob_module.glob(str(self.root / pattern), recursive=True)
            if Path(path).is_file()
        )

    def grep(self, pattern: str, path: str = ".") -> list[str]:
        base = self._path(path)
        return [
            f"{file}:{line_number}:{line.rstrip()}"
            for file in base.rglob("*")
            if file.is_file()
            for line_number, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
            if pattern in line
        ]

    def bash(self, command: str, timeout: int = 30) -> dict[str, Any]:
        if not self.allow_bash:
            raise PermissionError("bash permission is disabled")
        result = subprocess.run(
            command, cwd=self.root, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    def git(self, *args: str) -> dict[str, Any]:
        return self.bash(shlex.join(["git", *args]))
