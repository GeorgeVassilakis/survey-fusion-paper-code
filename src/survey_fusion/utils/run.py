"""Runtime context management and logging utilities.

This module provides a context manager to capture standard output and
error streams to a log file while simultaneously displaying them on
the console.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
import re
from typing import TextIO

__all__ = ("RunContext",)


_EPOCH_LINE_RE = re.compile(r"^(?:Epoch\s+(\d+):|Step\s+\d+\s+\(epoch\s+(\d+)\):)")
_TQDM_LINE_RE = re.compile(r"^[A-Za-z].*%\|.*\|.*(?:epoch|it|step)/s")


class _EpochFilteredLog(TextIO):  # type: ignore[misc]
    """A stream wrapper that filters epoch lines for log output only."""

    def __init__(self, stream: TextIO, every: int = 100):
        self._stream = stream
        self._every = max(1, every)
        self._buffer = ""

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        for ch in data:
            if ch == "\r":
                # Carriage returns are used by tqdm to redraw a single line.
                # Discard the in-progress line so it doesn't end up in the log.
                self._buffer = ""
                continue
            if ch == "\n":
                if self._should_log_line(self._buffer):
                    self._stream.write(self._buffer + "\n")
                self._buffer = ""
                continue
            self._buffer += ch
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer:
            if self._should_log_line(self._buffer):
                self._stream.write(self._buffer)
            self._buffer = ""
        self._stream.flush()

    def _should_log_line(self, line: str) -> bool:
        if line and not line.strip():
            return False
        text = line.lstrip()
        if _TQDM_LINE_RE.match(text):
            return False
        match = _EPOCH_LINE_RE.match(text)
        if not match:
            return True
        epoch_text = match.group(1) if match.group(1) is not None else match.group(2)
        if epoch_text is None:
            return True
        epoch = int(epoch_text)
        return epoch % self._every == 0


class _Tee(TextIO):  # type: ignore[misc]
    """A file-like object that duplicates writes to multiple streams.

    Parameters
    ----------
    *streams : `TextIO`
        Variable number of stream objects to write to.
    """

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:  # type: ignore[override]
        """Write string to all streams.

        Parameters
        ----------
        data : `str`
            The data to write.

        Returns
        -------
        count : `int`
            The length of the data written.
        """
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        """Flush all streams."""
        for stream in self._streams:
            stream.flush()


class RunContext(contextlib.AbstractContextManager["RunContext"]):
    """Context manager to redirect stdout/stderr to a log file and console.

    This context manager captures `sys.stdout` and `sys.stderr`, directing
    output to both the original streams and a `run.log` file created in
    the specified directory.

    Parameters
    ----------
    run_dir : `pathlib.Path`
        Directory where the ``run.log`` file will be created.
    epoch_log_every : `int | None`
        Only log epoch summary lines every N epochs (e.g., 100). Set to
        ``None`` or ``0`` to disable filtering.
    """

    def __init__(self, run_dir: Path, *, epoch_log_every: int | None = 100):
        self.run_dir = run_dir
        self.epoch_log_every = epoch_log_every
        self._log_file: TextIO | None = None
        self._filtered_log: TextIO | None = None
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self.log_path = self.run_dir / "run.log"

    def __enter__(self) -> "RunContext":
        """Enter the runtime context.

        Creates the run directory if it doesn't exist, opens the log file,
        and installs the teeing stream redirectors.

        Returns
        -------
        ctx : `RunContext`
            The context manager instance.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        log_stream: TextIO = self._log_file
        if self.epoch_log_every and self.epoch_log_every > 0:
            self._filtered_log = _EpochFilteredLog(self._log_file, every=self.epoch_log_every)
            log_stream = self._filtered_log
        sys.stdout = _Tee(self._stdout, log_stream)
        sys.stderr = _Tee(self._stderr, log_stream)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool | None:
        """Exit the runtime context.

        Restores the original standard output and error streams and closes
        the log file.
        """
        if self._filtered_log is not None:
            self._filtered_log.flush()
        if self._log_file is not None:
            self._log_file.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return None
