from __future__ import annotations
import copy
import csv
import time
import numpy as np
import pandas as pd

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

            Fx, Fy = float(F_vec[0, i]), float(F_vec[1, i])

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


def save_V0(path, V0):
    """
    Save offset of voltage of today
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, V0=np.asarray(V0, dtype=float))


def load_V0(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"V0 file not found: {path}")
    data = np.load(path)
    return data["V0"]


def load_perimeter_xy(
    xlsx_path: str,
    sheet_name=0,          # 0 = first sheet
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    """
    Loads perimeter points from an Excel file with columns named 'x' and 'y'.
    Returns: (N,2) float array [[x1,y1],...]
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Ensure required columns exist (case-insensitive fallback)
    cols_lower = {c.lower(): c for c in df.columns}
    if x_col not in df.columns and x_col.lower() in cols_lower:
        x_col = cols_lower[x_col.lower()]
    if y_col not in df.columns and y_col.lower() in cols_lower:
        y_col = cols_lower[y_col.lower()]

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns '{x_col}' and '{y_col}'. Found: {list(df.columns)}")

    # Keep only numeric x,y rows
    pts = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()

    if len(pts) < 3:
        raise ValueError(f"Need at least 3 perimeter points to fit a circle. Got {len(pts)}.")

    return pts.to_numpy(dtype=float)


def export_training_csv(path_csv: str, Sprvsr, T=None):
    """
    Save one row per training step t.
    """
    path_csv = Path(path_csv)
    path_csv.parent.mkdir(parents=True, exist_ok=True)

    if T is None:
        T = int(Sprvsr.T)

    # ---- header ----
    header = ["t", "tip_x", "tip_y", "tip_angle [deg]"]

    header += ["upd_tip_x", "upd_tip_y", "upd_tip_angle [deg]"]

    # loss columns (Sprvsr.loss_in_t is (T, loss_size))
    header += ["loss_x", "loss_y", "loss_MSE"]

    # measured
    header += ["Fx_meas", "Fy_meas"]

    # desired
    header += ["Fx_des", "Fy_des"]

    # ---- write ----
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for t in range(T):
            row = [t, float(Sprvsr.pos_in_t[t, 0]), float(Sprvsr.pos_in_t[t, 1]),
                   float(Sprvsr.pos_in_t[t, 2])]

            row += [float(Sprvsr.pos_update_in_t[t, 0]), 
                    float(Sprvsr.pos_update_in_t[t, 1]),
                    float(Sprvsr.pos_update_in_t[t, 2])]

            row += [float(x) for x in Sprvsr.loss_in_t[t, :]]

            row += [float(Sprvsr.loss_MSE_in_t[t])]

            # measured force
            row += [float(Sprvsr.F_in_t[t, 0]),
                    float(Sprvsr.F_in_t[t, 1])]

            # desired force
            row += [float(Sprvsr.desired_F_in_t[t, 0]),
                    float(Sprvsr.desired_F_in_t[t, 1])]

            w.writerow(row)


def load_stress_strain(path: str, file_type: str = 'csv'):
    """
    
    inputs:
    path      - 
    file_type - 'csv' - file (created by my stress strain code in main.ipynb)
                'txt' - my translation of Leon's shims experiments

    """
    if file_type == 'csv':
        df = pd.read_csv(path)

        thetas_vec = df["theta"].to_numpy(dtype=float)
        Fx_vec = df["F_x"].to_numpy(dtype=float)
        Fy_vec = df["F_y"].to_numpy(dtype=float)
        torque_x = df["torque_x"].to_numpy(dtype=float)
        torque_y = df["torque_y"].to_numpy(dtype=float)
    elif file_type == 'txt':
        data = np.loadtxt(path, delimiter=",")
        
        thetas_vec = data[:, 0]
        torque_x = data[:, 1]
        
        # dump other sizes as zeros
        torque_y = np.zeros(np.shape(torque_x))
        Fx_vec = np.zeros(np.shape(torque_x))
        Fy_vec = np.zeros(np.shape(torque_x))

    return thetas_vec, Fx_vec, Fy_vec, torque_x, torque_y
