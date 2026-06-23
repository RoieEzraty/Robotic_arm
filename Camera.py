from __future__ import annotations

import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import pyautogui  # NX Tether related

import re
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
                 width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.camera_id = int(CFG.Camera.camera_id)
        self.fps = float(CFG.Camera.fps)
        self.dt = 1.0 / self.fps

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if out_path is None:
            self.out_dir = Path("videos") / f"training_frames_{ts}"
        else:
            out_path = Path(out_path)
            self.out_dir = Path("videos") / f"{out_path.name}_{ts}"

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.width = int(CFG.Camera.width) if width is None else int(width)
        self.height = int(CFG.Camera.height) if height is None else int(height)

        # self.cap: Optional[cv2.VideoCapture] = None
        # self.thread: Optional[threading.Thread] = None
        # self.stop_event = threading.Event()
        # self.frame_i = 0

        self.mouse_x = int(CFG.Camera.NX_button_x)
        self.mouse_y = int(CFG.Camera.NX_button_y)

    def bring_nx_tether_to_front(self, timeout_s: float = 3.0) -> None:
        """Bring NX Tether window to foreground before clicking."""
        t0 = time.time()

        while time.time() - t0 < timeout_s:
            windows = pyautogui.getWindowsWithTitle("Z 7_2")

            if windows:
                win = windows[0]

                if win.isMinimized:
                    win.restore()
                    time.sleep(0.3)

                try:
                    win.activate()
                    time.sleep(0.3)
                except Exception as err:
                    print(f"NX activate warning: {err}")

                    # fallback: click inside the NX window to focus it
                    try:
                        win.restore()
                        time.sleep(0.2)
                        pyautogui.click(win.left + 50, win.top + 20)
                        time.sleep(0.3)
                    except Exception as err2:
                        print(f"NX fallback-focus warning: {err2}")

                return

            time.sleep(0.2)

        raise RuntimeError("Could not find NX Tether window. Open NX Tether before starting acquisition.")

    def click_nx_interval_button(self) -> None:
        self.bring_nx_tether_to_front()
        pyautogui.click(self.mouse_x, self.mouse_y)
        time.sleep(0.5)

    def natural_key(self, path: Path):
        """Sort filenames like image_2.jpg before image_10.jpg."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]

    def jpg_folder_to_video(self, frames_dir: str | Path, out_path: str | Path = "videos/from_jpg_folder.mp4",
                            fps: float = 5.0, sort_by: str = "name",   # "name" or "mtime"
                            resize_to_first: bool = True) -> Path:
        """Compile jpg/jpeg images from a folder into an mp4 video."""

        frames_dir = Path(frames_dir)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            image_paths.extend(frames_dir.glob(ext))

        if sort_by == "name":
            image_paths = sorted(image_paths, key=self.natural_key)
        elif sort_by == "mtime":
            image_paths = sorted(image_paths, key=lambda p: p.stat().st_mtime)
        else:
            raise ValueError("sort_by must be 'name' or 'mtime'.")

        print(f"Found {len(image_paths)} jpg/jpeg images in:")
        print(frames_dir)

        if len(image_paths) == 0:
            raise RuntimeError(f"No jpg/jpeg images found in {frames_dir}")

        print("First 5 frames:")
        for p in image_paths[:5]:
            print(" ", p.name)

        print("Last 5 frames:")
        for p in image_paths[-5:]:
            print(" ", p.name)

        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise RuntimeError(f"Could not read first image: {image_paths[0]}")

        h, w = first.shape[:2]
        print(f"Video size: {w} x {h}")
        print(f"fps: {fps}")

        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")

        written = 0
        skipped = 0

        try:
            for path in image_paths:
                frame = cv2.imread(str(path))

                if frame is None:
                    print(f"Skipping unreadable image: {path.name}")
                    skipped += 1
                    continue

                if frame.shape[:2] != (h, w):
                    if resize_to_first:
                        frame = cv2.resize(frame, (w, h))
                    else:
                        print(f"Skipping size-mismatch image: {path.name}, shape={frame.shape}")
                        skipped += 1
                        continue

                writer.write(frame)
                written += 1

        finally:
            writer.release()

        if written == 0:
            raise RuntimeError("No frames were written.")

        print(f"Written frames: {written}")
        print(f"Skipped frames: {skipped}")
        print(f"Saved video to: {out_path.resolve()}")

        return out_path


## ==========
## NOT IN USE
## ==========

    # def start(self) -> None:
    #     """Open camera and start background frame saving."""
    #     self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

    #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #     self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    #     ok, frame = self.cap.read()
    #     print(ok, frame.shape if ok else None)

    #     # self.cap.release()

    #     if self.width is not None:
    #         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    #     if self.height is not None:
    #         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    #     ok, _ = self.cap.read()
    #     if not ok:
    #         self.cap.release()
    #         self.cap = None
    #         raise RuntimeError(f"Could not read from camera_id={self.camera_id}.")

    #     self.stop_event.clear()
    #     self.thread = threading.Thread(target=self._loop, daemon=True)
    #     self.thread.start()

    # def stop(self, compile_video: bool = True, video_fps: Optional[float] = None) -> None:
    #     """Stop recording and release camera."""
    #     self.stop_event.set()

    #     if self.thread is not None:
    #         self.thread.join(timeout=3.0)

    #     if self.cap is not None:
    #         self.cap.release()

    #     self.thread = None
    #     self.cap = None

    #     if compile_video:
    #         return self.compile_video(video_fps=video_fps)

    # def _loop(self) -> None:
    #     """Save frames at approximately ``self.fps``."""
    #     assert self.cap is not None

    #     next_t = time.time()

    #     while not self.stop_event.is_set():
    #         ok, frame = self.cap.read()

    #         if ok:
    #             t_unix = time.time()
    #             filename = self.out_dir / f"frame_{self.frame_i:05d}_t_{t_unix:.3f}.png"
    #             cv2.imwrite(str(filename), frame)
    #             self.frame_i += 1

    #         next_t += self.dt
    #         sleep_t = next_t - time.time()
    #         if sleep_t > 0:
    #             time.sleep(sleep_t)

    # def compile_video(self, video_fps: Optional[float] = None) -> Path:
    #     """Compile saved PNG frames into an mp4 video."""
    #     frame_paths = sorted(self.out_dir.glob("frame_*.png"))
    #     if len(frame_paths) == 0:
    #         raise RuntimeError(f"No frames found in {self.out_dir}")

    #     if video_fps is None:
    #         video_fps = self.fps

    #     out_path = self.out_dir.with_suffix(".mp4")

    #     first = cv2.imread(str(frame_paths[0]))
    #     if first is None:
    #         raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    #     h, w = first.shape[:2]
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     writer = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), (w, h))

    #     try:
    #         for frame_path in frame_paths:
    #             frame = cv2.imread(str(frame_path))
    #             if frame is None:
    #                 print(f"Skipping unreadable frame: {frame_path}")
    #                 continue

    #             if frame.shape[:2] != (h, w):
    #                 frame = cv2.resize(frame, (w, h))

    #             writer.write(frame)

    #     finally:
    #         writer.release()

    #     return out_path

    # def nx_frames_to_video(self, frames_dir, out_path, fps=5.0):
    #     frames_dir = Path(frames_dir)
    #     out_path = Path(out_path)

    #     image_paths = sorted(
    #         list(frames_dir.glob("*.jpg")) +
    #         list(frames_dir.glob("*.jpeg")) +
    #         list(frames_dir.glob("*.JPG")) +
    #         list(frames_dir.glob("*.JPEG"))
    #     )

    #     if not image_paths:
    #         raise RuntimeError(f"No images found in {frames_dir}")

    #     first = cv2.imread(str(image_paths[0]))
    #     h, w = first.shape[:2]

    #     writer = cv2.VideoWriter(
    #         str(out_path),
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         fps,
    #         (w, h),
    #     )

    #     for path in image_paths:
    #         frame = cv2.imread(str(path))
    #         writer.write(frame)

    #     writer.release()
    #     return out_path
