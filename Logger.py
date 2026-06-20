import csv
import time
import sys
import traceback

from pathlib import Path
from typing import Iterable, Tuple, Optional
from datetime import datetime
from pathlib import Path

Vec3 = Tuple[float, float, float]


class Logger:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else self._timestamped_path()
        self._fh = None
        self._writer = None

    def start(self) -> None:
        is_new = not self.path.exists()
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if is_new:
            self._writer.writerow(
                ["t_unix", "pos_x", "pos_y", "pos_z", "force_x", "force_y"]
            )
            self._fh.flush()

    def stop(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
        self._fh = None
        self._writer = None

    def log(self, pos: Vec3, force: Vec3, t_unix: Optional[float] = None) -> None:
        if self._writer is None:
            self.start()
        t = time.time() if t_unix is None else t_unix
        self._writer.writerow([t, *pos, *force])
        self._fh.flush()

    def _timestamped_path(base_dir: str = "logs", prefix: str = "pos_force", ext: str = ".csv") -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return Path(base_dir) / f"{prefix}_{ts}{ext}"


class Tee:
    """Write text to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class NotebookOutputLogger:
    """Log notebook stdout/stderr to a text file while still showing output live."""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        base_dir: str = "logs",
        prefix: str = "training_output",
        ext: str = ".log",
    ):
        self.path = Path(path) if path else self._timestamped_path(base_dir, prefix, ext)
        self._fh = None
        self._old_stdout = None
        self._old_stderr = None

    def start(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._fh = self.path.open("w", encoding="utf-8")
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        sys.stdout = Tee(self._old_stdout, self._fh)
        sys.stderr = Tee(self._old_stderr, self._fh)

        print(f"Logging notebook output to: {self.path.resolve()}")
        return self.path

    def stop(self) -> None:
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr

        if self._fh is not None:
            self._fh.flush()
            self._fh.close()

        self._fh = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self) -> Path:
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None and self._fh is not None:
            self._fh.write("\n\n--- EXCEPTION TRACEBACK ---\n")
            self._fh.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            self._fh.flush()

        self.stop()

        # still show the error normally in Jupyter
        return False

    @staticmethod
    def _timestamped_path(base_dir: str, prefix: str, ext: str) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return Path(base_dir) / f"{prefix}_{ts}{ext}"
