import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.typing import NDArray


def fit_force_vs_voltage(voltages_array: np.ndarray, forces_array: np.ndarray,
                         std_array: np.ndarray | None = None) -> NDArray[np.float64]:
    """
    Fit F = a*V + b separately for x/y/z.

    Inputs:
        voltages_array: (N,3)
        forces_array:   (N,3)
        stds_voltage_array: (N,3) or None

    Returns:
        fit_params_array: (2,3) where:
            fit_params_array[0, :] = [ax, ay, az]
            fit_params_array[1, :] = [bx, by, bz]
    """
    fit_params = np.zeros((2, 3), dtype=float)
    for dim in range(3):
        if std_array is None:
            # ordinary least squares
            a, b = poly.polyfit(voltages_array[:, dim], forces_array[:, dim], deg=1)           
        else:
            sV = np.asarray(std_array[:, dim], dtype=float)
            # avoid divide-by-zero
            sV = np.where(sV <= 0, np.nanmedian(sV[sV > 0]), sV)
            w = 1.0 / (sV**2)
            a, b = poly.polyfit(voltages_array[:, dim], forces_array[:, dim], deg=1, w=w)
        fit_params[:, dim] = np.asarray(np.array([a, b]), dtype=float).reshape(2,)
    return fit_params
