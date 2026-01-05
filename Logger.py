import csv
import time
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
