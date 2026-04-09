from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# -------------------------------
# Saves/Writes
# -------------------------------
def write_supervisor_dataset(x_y_theta: NDArray[np.float64], F_vec: NDArray[np.float64], 
                             out_path: Optional[str | Path] = None, append: bool = False) -> Path:
    """Write supervisor dataset CSV compatible with :func:`load_pos_force`.
    Used in main Meca500 notebook

    Parameters
    ----------
    x_y_theta : NDArray[np.float64]. Tip positions of shape ``(N, 3)`` with columns ``[x, y, theta_z_deg]``.
    F_vec     : NDArray[np.float64]. Measured forces of shape ``(N, 2)`` with columns ``[Fx, Fy]``.
    out_path  : str | Path | None, optional. Output CSV path. If omitted, creates timestamped file under ``data/``.
    append    : bool, optional. If ``True``, append rows to an existing file.

    Returns
    -------
    Path to written CSV file.

    Notes
    -----
    Columns written are: ``t_unix, x_tip, y_tip, tip_angle_deg, F_x, F_y``
    """
    # initializations and cautions
    positions = np.asarray(x_y_theta, dtype=float)
    forces = np.asarray(F_vec, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"x_y_theta must have shape (N, 3). Got {positions.shape}.")
    if forces.ndim != 2 or forces.shape[1] != 2:
        raise ValueError(f"F_vec must have shape (N, 2). Got {forces.shape}.")
    if positions.shape[0] != forces.shape[0]:
        raise ValueError(f"x_y_theta and F_vec must have the same number of rows. "
                         f"Got {positions.shape[0]} and {forces.shape[0]}.")

    # create output file
    if out_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path("data").mkdir(parents=True, exist_ok=True)
        out_file = Path("data") / f"dataset_{timestamp}.csv"
    else:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    write_header = (mode == "w") or (not out_file.exists())

    # insert into file
    with out_file.open(mode, newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)

        # header
        if write_header:
            writer.writerow(["t_unix", "x_tip", "y_tip", "tip_angle_deg", "F_x", "F_y"])

        # insert data
        for i in range(positions.shape[0]):
            writer.writerow([time.time(), float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]),
                             float(forces[i, 0]), float(forces[i, 1])])
            file_obj.flush()

    # return the path, the file itself already written
    return out_file


def save_calibration_csv(path: str | Path, voltages_arr: NDArray[np.float64], forces_arr: NDArray[np.float64],
                         stds_arr: NDArray[np.float64]) -> None:
    """Save tri-axial calibration data to CSV.
    used in experiments.calibrate_forces_all_axes()

    Parameters
    ----------
    path         : str | Path. Output CSV path.
    voltages_arr : array. Voltage means of shape ``(N, 3)``.
    forces_arr   : array.  Applied forces of shape ``(N, 3)``.
    stds_arr     : array. Voltage standard deviations of shape ``(N, 3)``.
    """
    # caution and raises
    voltages = np.asarray(voltages_arr, dtype=float)
    forces = np.asarray(forces_arr, dtype=float)
    stds = np.asarray(stds_arr, dtype=float)

    if voltages.shape != forces.shape or voltages.shape != stds.shape:
        raise ValueError("voltages_arr, forces_arr, and stds_arr must all have the same shape.")
    if voltages.ndim != 2 or voltages.shape[1] != 3:
        raise ValueError(f"Expected arrays of shape (N, 3). Got {voltages.shape}.")

    # create output path
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # insert to file
    with out_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["Vx_V", "Vy_V", "Vz_V", "Fx_N", "Fy_N", "Fz_N", "stdVx_V", "stdVy_V", "stdVz_V"])
        for i in range(voltages.shape[0]):
            writer.writerow([float(voltages[i, 0]), float(voltages[i, 1]), float(voltages[i, 2]),
                             float(forces[i, 0]), float(forces[i, 1]), float(forces[i, 2]),
                             float(stds[i, 0]), float(stds[i, 1]), float(stds[i, 2])])


def save_V0(path: str | Path, V0: NDArray[np.float64] | list[float] | tuple[float, ...]) -> None:
    """Save zero-force voltage offset array to ``.npz``.
    Used in ForsentekClass.calibrate_daily()

    Parameters
    ----------
    path : str | Path. Output file path.
    V0   : array-like. Zero-force voltage offset.
    """
    # make output file
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save numpy array
    np.savez(out_path, V0=np.asarray(V0, dtype=float))


def export_training_csv(path_csv: str | Path, Sprvsr: object, T: Optional[int] = None) -> None:
    """Export predetermined-training data from supervisor buffers.
    Used at the end of Training in main Meca500 notebook.

    Parameters
    ----------
    path_csv : str | Path. Output CSV path.
    Sprvsr   : object. Supervisor-like object exposing ``pos_in_t``, ``pos_update_in_t``, ``loss_in_t``, 
                                                       ``loss_MSE_in_t``, ``F_in_t``, and ``desired_F_in_t``.
    T        : int | None, optional. Number of rows to export. If omitted, uses ``Sprvsr.T``.
    """
    # create output file
    out_path = Path(path_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # time steps to same
    if T is None:
        T = int(Sprvsr.T)

    # Headers
    header = ["t", "x_tip", "y_tip", "tip_angle_deg"]
    header += ["upd_x_tip", "upd_y_tip", "upd_tip_angle"]
    header += ["loss_x", "loss_y", "loss_MSE"]
    header += ["F_x_meas", "F_y_meas"]
    header += ["F_x_des", "F_y_des"]

    # insert
    with out_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)

        for t in range(T):
            row = [t, float(Sprvsr.pos_in_t[t, 0]), float(Sprvsr.pos_in_t[t, 1]), float(Sprvsr.pos_in_t[t, 2])]
            row += [float(Sprvsr.pos_update_in_t[t, 0]), float(Sprvsr.pos_update_in_t[t, 1]),
                    float(Sprvsr.pos_update_in_t[t, 2])]
            row += [float(x) for x in Sprvsr.loss_in_t[t, :]]
            row += [float(Sprvsr.loss_MSE_in_t[t])]
            row += [float(Sprvsr.F_in_t[t, 0]), float(Sprvsr.F_in_t[t, 1])]
            row += [float(Sprvsr.desired_F_in_t[t, 0]), float(Sprvsr.desired_F_in_t[t, 1])]
            writer.writerow(row)


def save_stress_strain(out_path: str | Path, thetas_vec: NDArray[np.float64], Fx_vec: NDArray[np.float64],
                       Fy_vec: NDArray[np.float64], torque_x: NDArray[np.float64],
                       torque_y: NDArray[np.float64]) -> None:
    """Save stress-strain measurement arrays to CSV.
    Used at end of stress-strain block in main Meca500 notebook

    Parameters
    ----------
    out_path                                       : str | Path. Output CSV path.
    thetas_vec, Fx_vec, Fy_vec, torque_x, torque_y : NDArray[np.float64]. Per-sample arrays of equal length.
    """
    # caution and raises
    thetas = np.asarray(thetas_vec, dtype=float)
    Fx = np.asarray(Fx_vec, dtype=float)
    Fy = np.asarray(Fy_vec, dtype=float)
    tau_x = np.asarray(torque_x, dtype=float)
    tau_y = np.asarray(torque_y, dtype=float)

    n = thetas.shape[0]
    if not all(arr.shape == (n,) for arr in (Fx, Fy, tau_x, tau_y)):
        raise ValueError("All input arrays must be 1D and have the same length.")

    # create output file
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # insert in file
    with out_file.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["theta", "F_x", "F_y", "torque_x", "torque_y"])

        for i in range(n):
            writer.writerow([thetas[i], Fx[i], Fy[i], tau_x[i], tau_y[i]])
            file_obj.flush()


# -------------------------------
# Loads
# -------------------------------
def load_pos_force(path: str | Path) -> list[dict[str, object]]:
    """Load position-force dataset rows from CSV.
    used in experiments.sweep_measurement_fixed_origami() and also in plot inside main Meca500 notebook.

    Supported schemas
    -----------------
    1. Measurement/training dataset:
       columns compatible with
       ``x_tip, y_tip, tip_angle_deg, F_x, F_y``
       or equivalent aliases.

       Returns rows of the form:
       ``{"pos": (x, y, theta_deg), "force": (Fx, Fy), "t_unix": ...}``

    2. Predetermined-training dataset:
       columns including
       ``upd_x_tip, upd_y_tip, upd_tip_angle, Fx_meas, Fy_meas, Fx_des, Fy_des``.

       Returns rows of the form:
       ``{"pos_meas": ..., "force_meas": ..., "pos_update": ..., "force_des": ..., "t_unix": ...}``

    Parameters
    ----------
    path : str | Path. Path to CSV file.

    Returns
    -------
    list[dict[str, object]]. Parsed dataset rows.
    """
    # initialize rows
    rows: list[dict[str, object]] = []

    # go over file
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        # read file
        reader = csv.DictReader(file_obj)

        if reader.fieldnames is None:
            return rows

        # booleans. These are optional data in file
        has_time = "t_unix" in reader.fieldnames
        has_deg = "tip_angle_deg" in reader.fieldnames
        has_rad = "tip_angle_rad" in reader.fieldnames
        has_update = "upd_x_tip" in reader.fieldnames

        for record in reader:
            x = _get_first_in_file(record, ["pos_x", "x_tip", "Px"], name="x")
            y = _get_first_in_file(record, ["pos_y", "y_tip", "Py"], name="y")

            if has_deg and record["tip_angle_deg"] != "":
                theta_deg = float(record["tip_angle_deg"])
            elif has_rad and record["tip_angle_rad"] != "":
                theta_deg = float(np.rad2deg(float(record["tip_angle_rad"])))
            else:
                raise ValueError(
                    "File must contain either 'tip_angle_deg' or 'tip_angle_rad'.",
                )

            if has_update:
                row: dict[str, object] = {
                    "pos_meas": (x, y, theta_deg),
                    "force_meas": (float(record["Fx_meas"]), float(record["Fy_meas"])),
                    "pos_update": (
                        float(record["upd_x_tip"]),
                        float(record["upd_y_tip"]),
                        float(record["upd_tip_angle"]),
                    ),
                    "force_des": (float(record["Fx_des"]), float(record["Fy_des"])),
                }
            else:
                Fx = _get_first_in_file(record, ["force_x", "F_x", "Fx"], name="Fx")
                Fy = _get_first_in_file(record, ["force_y", "F_y", "Fy"], name="Fy")
                row = {
                    "pos": (x, y, theta_deg),
                    "force": (Fx, Fy),
                }

            if has_time and record["t_unix"] != "":
                row["t_unix"] = float(record["t_unix"])
            else:
                row["t_unix"] = None

            rows.append(row)

    return rows


def load_calibration_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load tri-axial calibration data from CSV.
    Used in ForsentekClass.calibrate_daily()

    Parameters
    ----------
    path : str | Path. Calibration CSV path.

    Returns
    -------
    Voltage means, forces, voltage standard deviations, each of shape ``(N, 3)``.
    """
    # initialize
    voltages: list[list[float]] = []
    forces: list[list[float]] = []
    stds: list[list[float]] = []

    # read file from path
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            voltages.append([float(row["Vx_V"]), float(row["Vy_V"]), float(row["Vz_V"])])
            forces.append([float(row["Fx_N"]), float(row["Fy_N"]), float(row["Fz_N"])])
            stds.append([float(row["stdVx_V"]), float(row["stdVy_V"]), float(row["stdVz_V"])])

    return (np.asarray(voltages, dtype=float), np.asarray(forces, dtype=float), np.asarray(stds, dtype=float))


def load_V0(path: str | Path) -> NDArray[np.float64]:
    """Load zero-force voltage offset array from ``.npz``.
    Used in ForsentekClass.calibrate_daily()

    Parameters
    ----------
    path : str | Path. Input file path.

    Returns
    -------
    Loaded voltage-offset vector. size (3,)
    """
    # caution and raises
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"V0 file not found: {in_path}")

    # load and return
    data = np.load(in_path)
    return np.asarray(data["V0"], dtype=float)


def load_perimeter_xy(xlsx_path: str | Path, sheet_name: int | str = 0,
                      x_col: str = "x", y_col: str = "y") -> NDArray[np.float64]:
    """Load planar perimeter points from Excel file.
    Used in initiation of MecaClass, for calculation of R_robot

    Parameters
    ----------
    xlsx_path    : str | Path. Excel file path.
    sheet_name   : int | str, optional. Excel sheet index or name.
    x_col, y_col : str, optional. Column names for x and y values. Case-insensitive fallback applied.

    Returns
    -------
    Array, size ``(N, 2)``, containing perimeter points.
    """
    # read as dataframe
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    cols_lower = {str(c).lower(): c for c in df.columns}
    if x_col not in df.columns and x_col.lower() in cols_lower:
        x_col = str(cols_lower[x_col.lower()])
    if y_col not in df.columns and y_col.lower() in cols_lower:
        y_col = str(cols_lower[y_col.lower()])

    # raises
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns '{x_col}' and '{y_col}'. Found: {list(df.columns)}")

    # points
    points = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(points) < 3:
        raise ValueError(f"Need at least 3 perimeter points to fit a circle. Got {len(points)}.")

    return points.to_numpy(dtype=float)


def load_stress_strain(path: str | Path, file_type: str = "csv") -> tuple[NDArray[np.float64], NDArray[np.float64],
                                                                          NDArray[np.float64], NDArray[np.float64],
                                                                          NDArray[np.float64]]:
    """Load stress-strain data from CSV or translated TXT format.
    Used sporadically for plots in main Meca500 notebook


    Parameters
    ----------
    path      : str | Path. Input stress-strain path.
    file_type : {"csv", "txt"}
                 ``"csv"`` for files saved by :func:`save_stress_strain`,
                 ``"txt"`` for the translated shim-measurement text format.

    Returns
    -------
    thetas_vec, Fx_vec, Fy_vec, torque_x, torque_y : arrays of force and torque for every angle
    """
    if file_type == "csv":
        df = pd.read_csv(path)
        thetas_vec = df["theta"].to_numpy(dtype=float)
        Fx_vec = df["F_x"].to_numpy(dtype=float)
        Fy_vec = df["F_y"].to_numpy(dtype=float)
        torque_x = df["torque_x"].to_numpy(dtype=float)
        torque_y = df["torque_y"].to_numpy(dtype=float)
    elif file_type == "txt":
        data = np.loadtxt(path, delimiter=",")
        thetas_vec = np.asarray(data[:, 0], dtype=float)
        torque_x = np.asarray(data[:, 1], dtype=float)
        torque_y = np.zeros_like(torque_x)
        Fx_vec = np.zeros_like(torque_x)
        Fy_vec = np.zeros_like(torque_x)
    else:
        raise ValueError("file_type must be either 'csv' or 'txt'.")

    return thetas_vec, Fx_vec, Fy_vec, torque_x, torque_y


def build_torque_from_file(path: str | Path, *, 
                           angles_in_degrees: bool = True,) -> Callable[[NDArray[np.float64] | float],
                                                                        NDArray[np.float64] | float]:
    """Build a clamped torque-vs-angle interpolation function from file.
    Used to build tau_of_theta in init of ForsentekClass

    Parameters
    ----------
    path              : str | Path. Path to a two-column text or CSV file containing angle and torque.
    angles_in_degrees : bool, optional

    Returns
    -------
    Callable function mapping angle query values to linearly interpolated torque values, 
    clamped to the measured angle range.

    Notes
    -----
    - Duplicate angle samples are collapsed by keeping the first occurrence.
    - Some known historical files require torque sign inversion and are handled
      explicitly to preserve current project behavior.
    """
    # load with caution
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, delimiter=",")
    theta = np.asarray(data[:, 0], dtype=float)
    tau = np.asarray(data[:, 1], dtype=float)

    # some files need inversion of torque
    if str(path) in {"single_hinge_files/Roie_metal_singleMylar_short.csv",
                     "single_hinge_files/Stress_Strain_steel_1myl1tp_short.csv",
                     "single_hinge_files/Stress_Strain_1myl1tp_otherEnd_short.csv"}:
        tau = -tau

    # sort
    order = np.argsort(theta)
    theta = theta[order]
    tau = tau[order]

    # use unique values
    theta_unique, idx = np.unique(theta, return_index=True)
    tau_unique = tau[idx]

    # discrete values of angle and torque
    theta_grid = np.asarray(theta_unique, dtype=np.float32)
    torque_grid = np.asarray(tau_unique, dtype=np.float32)

    # continuous function
    def torque_of_theta(theta_query: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        theta_query_arr = np.asarray(theta_query, dtype=float)
        theta_clamped = np.clip(theta_query_arr, theta_grid[0], theta_grid[-1])
        torque_interp = np.interp(theta_clamped, theta_grid, torque_grid)

        if np.ndim(theta_query) == 0:
            return float(torque_interp)
        return np.asarray(torque_interp, dtype=float)

    return torque_of_theta


def _get_first_in_file(r: Mapping[str, Union[str, float, int, None]], keys: Iterable[str], *,
                       name: str = "", allow_missing: bool = False) -> Optional[float]:
    """Extract the first valid scalar value from a row-like mapping.
    Used in load_pos_force()

    Parameters
    ----------
    r             : Mapping[str, str | float | int | None]. Record-like object, typically a CSV row dictionary.
    keys          : Iterable[str]. Ordered candidate keys to search for.
    name          : str, optional. Human-readable field name used only in error reporting.
    allow_missing : bool, optional. If ``True``, return ``None`` when no valid key is found.

    Returns
    -------
    float | None. First successfully parsed scalar value.
    """
    for key in keys:
        if key in r and r[key] not in ("", None):
            return float(r[key])

    # return None if not found
    if allow_missing:
        return None

    raise KeyError(f"None of {list(keys)} found for {name}")
