import copy
import csv


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