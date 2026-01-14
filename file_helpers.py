import copy
import csv
import numpy as np


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
