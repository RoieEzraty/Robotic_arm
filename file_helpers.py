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


def save_calibration_csv(path, voltages_vec, weights_vec, stds_vec):
    assert len(voltages_vec) == len(weights_vec), "Length mismatch"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["voltage_V", "weight_N", "std"])  # header
        for v, w, s in zip(voltages_vec, weights_vec, stds_vec):
            writer.writerow([v, w, s])


def load_calibration_csv(path):
    voltages = []
    forces = []
    stds = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            voltages.append(float(row["voltage_V"]))
            forces.append(float(row["weight_N"]))
            stds.append(float(row["std"]))

    return np.asarray(voltages), np.asarray(forces), np.asarray(stds)
