#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, List, Sequence


class ReadwiseConnectorError(RuntimeError):
    """Base connector error."""


class ReadwiseCliMissingError(ReadwiseConnectorError):
    pass


class ReadwiseCommandError(ReadwiseConnectorError):
    def __init__(self, message: str, *, command: Sequence[str], exit_code: int, stdout: str, stderr: str):
        super().__init__(message)
        self.command = list(command)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class ReadwiseJsonError(ReadwiseConnectorError):
    pass


@dataclass
class CommandResult:
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    data: Any


class ReadwiseConnector:
    def __init__(self, binary: str = "readwise", timeout_seconds: int = 60):
        self.binary = binary
        self.timeout_seconds = timeout_seconds

    def ensure_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise ReadwiseCliMissingError(
                f"Readwise CLI '{self.binary}' was not found on PATH."
            )

    def run_json(self, args: Sequence[str]) -> CommandResult:
        self.ensure_available()
        command = [self.binary, "--json", *args]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReadwiseCommandError(
                f"Readwise command timed out after {self.timeout_seconds}s.",
                command=command,
                exit_code=-1,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            ) from exc

        if completed.returncode != 0:
            raise ReadwiseCommandError(
                self._classify_failure(completed.stderr or completed.stdout),
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ReadwiseJsonError(
                f"Readwise CLI returned non-JSON output for command: {' '.join(command)}"
            ) from exc

        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            data=data,
        )

    @staticmethod
    def _classify_failure(raw_error: str) -> str:
        message = (raw_error or "").strip()
        lowered = message.lower()
        if "not authenticated" in lowered or "login" in lowered or "oauth" in lowered:
            return "Readwise CLI appears unauthenticated."
        if "unknown option" in lowered or "missing required" in lowered or "usage:" in lowered:
            return "Readwise CLI usage error."
        if "network" in lowered or "timed out" in lowered or "econn" in lowered:
            return "Readwise CLI/network failure."
        return message or "Readwise CLI command failed."
