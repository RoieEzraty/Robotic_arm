from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariabsConfig:
    ip: str = "192.168.0.100"

    # motion
    lin_vel: float = 20.0
    ang_vel: float = 10.0
    lin_acc: float = 20.0
    ang_acc: float = 20.0
    vel: float = 10.0
    acc: float = 20.0
    blending: int = 0

    # position / geometry
    load_cell_thick: float = 18.0
    holder_len: float = 19.0
    cable_holder_len: float = 7.0
    offset_chain_tip: float = 23.6
    pole_len_mid: float = 23.6
    pole_rad: float = 3.75

    # home of robot, motion is taken from there
    pos_home: tuple[float, float, float, float, float, float] = (189.0, 0.0, 28.0, 179.9, 0.1, 0.1)
    # pos_origin: tuple[float, float] = (83.2, -12.3)  # long table holder
    pos_origin: tuple[float, float] = (83.2 - 26, -12.3)  # short table holder
    joints_sleep: tuple[float, float, float, float, float, float] = (0.0, -30.0, 20.0, 0.0, 100.0, 90.0)
    norm_length: float = 47.2
    norm_angle: float = 180.0
    theta_sim_to_robot: float = -1.0

    limits_path: str = r"data\calibrations\arm limits in x y.xlsx"
    tau_file: str = r"single_hinge_files\Mar12_dl90.csv"  # single hinge stress-strain


@dataclass(frozen=True)
class SprvsrConfig:
    experiment: str = "training"
    dataset_type: str = "from file"
    dataset_path: str = (
        r"data\measurements\Feb18\0001to1000_2\dataset.csv"
    )

    # sweep
    sweep_T: int = 12
    x_range: int = 100
    y_range: int = 100
    theta_range: int = 180

    # training
    T: int = 16
    rand_key_dataset: int = 16
    alpha: float = 0.4
    init_buckle: tuple[int, ...] = (1, 1, 1, 0)
    desired_buckle: tuple[int, ...] = (1, 1, 1, 1)

    # chain / files
    L: float = 47.2  # 45mm plastic edge + 1.2mm tape (each direction)
    H: int = 4
    convert_F: float = 1000.0  # N to mN

    # match simulation and experiment
    origin_rel_to_sim = [108.0, -14.0, 0.0]


@dataclass(frozen=True)
class SnsrConfig:
    DEV: str = "Dev1"
    Channel_x: str = "ai1"
    Channel_y: str = "ai2"
    Channel_z: str = "ai3"
    samp_freq: float = 200.0
    T: float = 0.5

    min_val: float = -5.0
    max_val: float = 5.0

    calibration_path: str = r"data\calibrations\calibration_triaxial_2026-02-09_13-55-56.csv"
    V0_path: str = r"data\calibrations\V0_latest.npz"

    angle: float = 90.0
    norm_force: float = 0.1


@dataclass(frozen=True)
class Config:
    Variabs: VariabsConfig = field(default_factory=VariabsConfig)
    Sprvsr: SprvsrConfig = field(default_factory=SprvsrConfig)
    Snsr: SnsrConfig = field(default_factory=SnsrConfig)


CFG = Config()
