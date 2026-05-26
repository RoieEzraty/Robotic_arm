from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariabsConfig:
    ip: str = "192.168.0.100"

    # motion
    # lin_vel: float = 20.0  # good 2026March
    # ang_vel: float = 10.0  # good 2026March
    # lin_vel: float = 15.0  # for force listener 2026Apr
    # ang_vel: float = 2.5  # for force listener 2026Apr
    # lin_acc: float = 20.0
    # ang_acc: float = 20.0
    joints_vel_update: float = 10.0  # joints
    joints_acc_update: float = 20.0  # joints

    # motion: slow / update / force-sensitive
    lin_vel_update: float = 12.0       # mm/s
    lin_acc_update: float = 12.0       # mm/s^2
    ang_vel_update: float = 10.0       # deg/s
    ang_acc_update: float = 10.0       # deg/s^2
    # acc_update: float = 20.0           # percent, affects both linear + angular accel

    # motion: fast / measurement sweep
    lin_vel_measurement: float = 35.0  # mm/s
    lin_acc_measurement: float = 35.0  # mm/s^2
    ang_vel_measurement: float = 10.0  # deg/s
    ang_acc_measurement: float = 10.0  # deg/s^2
    # acc_measurement: float = 20.0      # percent

    blending_update: int = 0
    blending_measurement: int = 0

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
    pos_origin: tuple[float, float, float] = (83.2 - 26, -12.3)  # short table holder
    pos_stress_strain: tuple[float, float, float] = (72, 0, 0)
    joints_sleep: tuple[float, float, float, float, float, float] = (0.0, -30.0, 20.0, 0.0, 100.0, 90.0)
    norm_length: float = 47.2
    norm_angle: float = 180.0
    theta_sim_to_robot: float = -1.0

    limits_path: str = r"data\calibrations\arm limits in x y.xlsx"  # 2d points maximal tip values, approx. on radius
    tau_file: str = r"single_hinge_files\Mar12_dl90.csv"  # single hinge stress-strain


@dataclass(frozen=True)
class SprvsrConfig:
    # experiment: str = "training"  # full training through robot
    experiment: str = "predetermined training"  # training done in simulation, just implement on robot
    # dataset_type: str = "from file"
    dataset_type: str = "predetermined"
    # dataset_type: str = "None"
    # dataset_path: str = r"data\measurements\Feb18\0001to1000_2\dataset.csv"
    dataset_path: str = r"data\datasets\May22\arc\{}.csv"

    # # BEASTAL update tip values
    # update_scheme: str = 'one_to_one'  # direct normalized loss, equal to num of outputs
    update_scheme: str = 'loss_diff'  # difference of x and y loss components
    # update_scheme: str = 'pos'  # difference of x and y loss components

    # normalize_step: bool = True
    normalize_step: bool = False

    # sweep
    sweep_T: int = 12
    x_range: int = 100
    y_range: int = 100
    theta_range: int = 180

    # training
    T: int = 16
    rand_key_dataset: int = 16
    alpha: float = 0.2  # [dimless]
    init_buckle: tuple[int, ...] = "0000"
    desired_buckle: tuple[int, ...] = "0001"

    # chain / files
    L: float = 47.2  # 45mm plastic edge + 1.2mm tape (each direction)
    H: int = 4
    convert_F: float = 1000.0  # N to mN

    # match simulation and experiment
    origin_rel_to_sim = [108.0, -14.0, 0.0]

    # reach zero-force for update_scheme == 'pos'
    xy_step_size: float = 10   # [mm]
    theta_step_size: float = 10  # [deg]
    F_tol: float = 5  # [mN]


@dataclass(frozen=True)
class SnsrConfig:
    DEV: str = "Dev1"
    Channel_x: str = "ai1"
    Channel_y: str = "ai2"
    Channel_z: str = "ai3"
    samp_freq: float = 200.0
    T: float = 1.0

    min_val: float = -5.0
    max_val: float = 5.0

    calibration_path: str = r"data\calibrations\calibration_triaxial_2026-05-21_14-07-18.csv"
    V0_path: str = r"data\calibrations\V0_latest.npz"

    angle: float = 90.0
    norm_force: float = 0.1

    force_threshold: float = 1.0
    force_chunk_T: float = 0.02
    revert_on_force: bool = True


@dataclass(frozen=True)
class Config:
    Variabs: VariabsConfig = field(default_factory=VariabsConfig)
    Sprvsr: SprvsrConfig = field(default_factory=SprvsrConfig)
    Snsr: SnsrConfig = field(default_factory=SnsrConfig)


CFG = Config()
