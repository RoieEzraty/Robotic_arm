import numpy as np
from numpy.typing import NDArray


def fit_force_vs_voltage(voltages_vec: np.ndarray,
                         forces_vec: np.ndarray,
                         stds_voltage: np.ndarray | None = None) -> NDArray[float]:
    """
    Fit: F = a*V + b

    If stds_voltage is provided (same length as voltages_vec),
    performs weighted least squares using sigma_F = a*sigma_V approximation
    is circular, so instead we weight by 1/sigma_V^2 as a practical proxy.
    """
    V = np.asarray(voltages_vec, dtype=float)
    F = np.asarray(forces_vec, dtype=float)

    if stds_voltage is None:
        # ordinary least squares
        a, b = np.polyfit(V, F, deg=1)
        return a, b

    sV = np.asarray(stds_voltage, dtype=float)
    # avoid divide-by-zero
    sV = np.where(sV <= 0, np.nanmedian(sV[sV > 0]), sV)
    w = 1.0 / (sV**2)

    a, b = np.polyfit(V, F, deg=1, w=w)
    return np.array([a, b])
