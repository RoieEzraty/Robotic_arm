from __future__ import annotations
import copy
import csv
import numpy as np
import time

from pathlib import Path
from datetime import datetime
from typing import Optional
from numpy.typing import NDArray


def load_pos_force(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "t_unix": float(r["t_unix"]),
                "pos": (float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])),
                "force": (float(r["force_x"]), float(r["force_y"])),
            })
    return rows


def write_supervisor_dataset(
    x_y_theta: NDArray[np.float64],  # shape (N, 3) with columns [x, y, theta_z_deg]
    F_vec: NDArray[np.float64],  # shape (N, 2) with columns [Fx, Fy]
    out_path: Optional[str | Path] = None,
    append: bool = False,
) -> Path:
    """
    Build a dataset CSV compatible with file_helpers.load_pos_force().

    CSV columns written:
        t_unix,pos_x,pos_y,pos_z,force_x,force_y

    """
    x_y_theta = np.asarray(x_y_theta, dtype=float)
    if x_y_theta.ndim != 2 or x_y_theta.shape[1] != 3:
        raise ValueError(f"x_y_theta must be (N,3). Got {x_y_theta.shape}")

    # default output path
    if out_path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path("data").mkdir(parents=True, exist_ok=True)
        out_path = Path("data") / f"dataset_{ts}.csv"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    write_header = (mode == "w") or (not out_path.exists())

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["t_unix", "pos_x", "pos_y", "pos_z", "force_x", "force_y"])

        for i in range(x_y_theta.shape[0]):
            x, y, theta_z_deg = (float(x_y_theta[i, 0]),
                                 float(x_y_theta[i, 1]),
                                 float(x_y_theta[i, 2]))

            Fx, Fy = float(F_vec[i, 0]), float(F_vec[i, 0])

            # ---- write row ----
            t_unix = time.time()
            writer.writerow([t_unix, x, y, theta_z_deg, Fx, Fy])
            f.flush()

    return out_path


def save_calibration_csv(path, voltages_arr, forces_arr, stds_arr):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Vx_V", "Vy_V", "Vz_V",
            "Fx_N", "Fy_N", "Fz_N",
            "stdVx_V", "stdVy_V", "stdVz_V"
        ])
        for i in range(voltages_arr.shape[0]):
            writer.writerow([
                voltages_arr[i, 0], voltages_arr[i, 1], voltages_arr[i, 2],
                forces_arr[i, 0], forces_arr[i, 1], forces_arr[i, 2],
                stds_arr[i, 0], stds_arr[i, 1], stds_arr[i, 2],
            ])


def load_calibration_csv(path):
    V, F, S = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            V.append([float(row["Vx_V"]), float(row["Vy_V"]), float(row["Vz_V"])])
            F.append([float(row["Fx_N"]), float(row["Fy_N"]), float(row["Fz_N"])])
            S.append([float(row["stdVx_V"]), float(row["stdVy_V"]), float(row["stdVz_V"])])

    return np.asarray(V, dtype=float), np.asarray(F, dtype=float), np.asarray(S, dtype=float)
    # return np.asarray(V), np.asarray(F), np.asarray(S)
