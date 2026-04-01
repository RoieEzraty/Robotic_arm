from __future__ import annotations
import copy
import csv
import time
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Callable, Union, Iterable, Mapping
from numpy.typing import NDArray


def load_pos_force(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        has_time = "t_unix" in reader.fieldnames
        has_deg = "tip_angle_deg" in reader.fieldnames
        has_rad = "tip_angle_rad" in reader.fieldnames
        has_update = "upd_x_tip" in reader.fieldnames

        for r in reader:
            x = _get_first_in_file(r, ["pos_x", "x_tip", "Px"], name="x")
            y = _get_first_in_file(r, ["pos_y", "y_tip", "Py"], name="y")
            # ----- angle handling -----
            if has_deg and r["tip_angle_deg"] != "":
                theta_deg = float(r["tip_angle_deg"])
            elif has_rad and r["tip_angle_rad"] != "":
                theta_deg = np.rad2deg(float(r["tip_angle_rad"]))
            else:
                raise ValueError(
                    "File must contain either 'tip_angle_deg' or 'tip_angle_rad'"
                )

            if has_update:
                row = {"pos_meas": (x, y, theta_deg),
                       "force_meas": (float(r["Fx_meas"]), float(r["Fy_meas"])),
                       "pos_update": (float(r["upd_x_tip"]), float(r["upd_y_tip"]), 
                                      float(r["upd_tip_angle"])),
                       "force_des": (float(r["Fx_des"]), float(r["Fy_des"]))}

            else:
                Fx = _get_first_in_file(r, ["force_x", "F_x", "Fx"], name="Fx")
                Fy = _get_first_in_file(r, ["force_y", "F_y", "Fy"], name="Fy")
                row = {"pos": (x, y, theta_deg),
                       "force": (Fx, Fy)}

            # ----- optional time -----
            if has_time and r["t_unix"] != "":
                row["t_unix"] = float(r["t_unix"])
            else:
                row["t_unix"] = None

            rows.append(row)

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
    x_y_theta = np.asarray(x_y_theta, dtype=float)  # [mm, mm, deg]
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
            writer.writerow(["t_unix", "x_tip", "y_tip", "tip_angle_deg", "F_x", "F_y"])

        for i in range(x_y_theta.shape[0]):
            x, y, theta_z_deg = (float(x_y_theta[i, 0]),
                                 float(x_y_theta[i, 1]),
                                 float(x_y_theta[i, 2]))

            Fx, Fy = float(F_vec[i, 0]), float(F_vec[i, 1])

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
    header = ["t", "x_tip", "y_tip", "tip_angle_deg"]

    header += ["upd_x_tip", "upd_y_tip", "upd_tip_angle_deg"]

    # loss columns (Sprvsr.loss_in_t is (T, loss_size))
    header += ["loss_x", "loss_y", "loss_MSE"]

    # measured
    header += ["F_x_meas", "F_y_meas"]

    # desired
    header += ["F_x_des", "F_y_des"]

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


def save_stress_strain(out_path, thetas_vec, Fx_vec, Fy_vec, torque_x, torque_y):
    out_path = Path(out_path)
    mode = "w"
    write_header = (mode == "w") or (not out_path.exists())

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["theta", "F_x", "F_y", "torque_x", "torque_y"])

        for i in range(thetas_vec.shape[0]):
            writer.writerow([thetas_vec[i], Fx_vec[i], Fy_vec[i], 
                             torque_x[i], torque_y[i]])
            f.flush()


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


# ---------------------------------------------------------------
# Build functions from file
# ---------------------------------------------------------------
def build_torque_from_file(path: str, *, angles_in_degrees: bool = True):
    """
    Load torque–angle measurements from file and construct JAX-compatible interpolation functions for torque and stiffness.
    Stiffness ``k = dτ/dθ``

    Parameters
    ----------
    path              - str, path to the text/CSV file containing two columns: angle and torque.
    contact           - bool, If True, extend torque function outside measured range to represent contact-induced divergence.
    angles_in_degrees - bool, If True, convert angles from degrees to radians.
    savgol_window     - Optional[int], window length for Savitzky–Golay smoothing of the stiffness curve. Must be odd integer > 2.
    contact_scale     - float, contact scaling factor relative to maximal measured torque.

    Returns
    -------
    theta_grid      - jnp.ndarray, shape (N,), sorted vector of angles (radians).
    torque_grid     - jnp.ndarray, shape (N,), torque samples over theta.
    k_grid          - jnp.ndarray, shape (N,), local stiffness as numeric derivative of torque w.r.t. theta.
    torque_of_theta - callable, JAX function theta -> torque interpolation including diverging forces at contact.
    k_of_theta      - callable, JAX function theta -> stiffness interpolation.

    Notes
    -----
    - Negative stiffness values clipped to small positive value (``1e-3``).
    - Contact occurs outside the measured range.
    """
    # ------ load as numpy, sort, unique------
    try:
        data = np.loadtxt(path)                # shape (N, 2)
    except ValueError:
        data = np.loadtxt(path, delimiter=',')
    theta = data[:, 0]
    tau = data[:, 1]

    if path in {"single_hinge_files/Roie_metal_singleMylar_short.csv", 
                "single_hinge_files/Stress_Strain_steel_1myl1tp_short.csv", 
                "single_hinge_files/Stress_Strain_1myl1tp_otherEnd_short.csv"}:  # flip axes
        tau = -tau

    # # degrees -> radians if needed
    # if angles_in_degrees:
    #     theta = np.deg2rad(theta)

    # sort & unique (interp requires monotonic x)
    order = np.argsort(theta)
    theta = theta[order]
    tau = tau[order]
    # collapse duplicates (if any)
    theta_u, idx = np.unique(theta, return_index=True)
    tau_u = tau[idx]

    # ------ arrays ------
    theta_grid = np.asarray(theta_u, dtype=np.float32)
    torque_grid = np.asarray(tau_u, dtype=np.float32)

    # ----- linear interpolators ------
    def torque_of_theta(theta_query: np.ndarray) -> np.ndarray:
        # masks for outside vs inside range
        th = _clamp(theta_query, theta_grid[0], theta_grid[-1])
        tau = np.interp(th, theta_grid, torque_grid)  # torque
        return tau

    def _clamp(x, xmin, xmax):
        return np.clip(x, xmin, xmax)

    return torque_of_theta


# ---------------------------------------------------------------
# Simple functions
# ---------------------------------------------------------------
def _get_first_in_file(r: Mapping[str, Union[str, float, int, None]], keys: Iterable[str], *, name: str = "",
                       allow_missing: bool = False) -> Optional[float]:
    """
    Extract first valid scalar value from a csv, using list of candidate keys. If no valid key is found: returns `None`

    Parameters
    ----------
    r             - Mapping[str, str | float | int | None]. Record-like object (e.g. CSV row dictionary).
    keys          - Iterable[str]. Ordered candidate keys to search for.
    name          - str, optional.  Human-readable field name used only for error reporting.
    allow_missing - bool, optional. If True, return None when no key is found.

    Returns
    -------
    value - float or None. First successfully parsed scalar value.
    """
    for k in keys:
        if k in r and r[k] not in ("", None):
            return float(r[k])
    if allow_missing:
        return None
    raise KeyError(f"None of {keys} found for {name}")
