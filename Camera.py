from __future__ import annotations

import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2


class Camera:
    """Record webcam video or frames during training.

    Parameters
    ----------
    camera_id : int
        OpenCV camera index. Usually 0 for built-in/default camera, 1 for external USB camera.
    fps : float
        Desired acquisition frequency [Hz].
    out_path : str | Path | None
        Output video path. If None, creates timestamped path under ``videos/``.
    width, height : int | None
        Optional camera resolution request.
    """

    def __init__(self, CFG: Optional[object] = None, out_path: Optional[str | Path] = None, 
                 width: Optional[int] = 1280, height: Optional[int] = 720) -> None:
        self.camera_id = int(CFG.Camera.camera_id)
        self.fps = float(CFG.Camera.fps)
        self.dt = 1.0 / self.fps

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path("videos").mkdir(parents=True, exist_ok=True)
        if out_path is None:
            self.out_path_w_ending = Path("videos") / f"training_video_{ts}.mp4"
        else:
            self.out_path_w_ending = Path("videos") / str(out_path + f"_{ts}.mp4")
            self.out_path_w_ending.parent.mkdir(parents=True, exist_ok=True)

        self.width = int(CFG.Camera.width)
        self.height = int(CFG.Camera.height)

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self, out_path: Optional[str] = None) -> None:
        """Open camera and start background recording."""
        if out_path:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_path_w_ending = Path("videos") / str(out_path + f"_{ts}.mp4")
        else:
            out_path_w_ending = self.out_path_w_ending


        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"Could not read from camera_id={self.camera_id}.")

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(out_path_w_ending), fourcc, self.fps, (w, h))

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop recording and release camera."""
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=3.0)

        if self.writer is not None:
            self.writer.release()
        if self.cap is not None:
            self.cap.release()

        self.thread = None
        self.writer = None
        self.cap = None

    def _loop(self) -> None:
        """Record frames at approximately ``self.fps``."""
        assert self.cap is not None
        assert self.writer is not None

        next_t = time.time()

        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if ok:
                self.writer.write(frame)

            next_t += self.dt
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)