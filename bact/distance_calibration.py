import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def load_calibration(json_file="calibration_data.json"):
    with open(json_file, "r") as f:
        data = json.load(f)
    # Convert keys to int, remove None values
    return {int(k): v for k, v in data.items() if v is not None}


def get_calibration_function_and_limits(calibration_json):
    calibration_dict = load_calibration(calibration_json)
    distances, radii = [], []
    min_rad_lin_area = 50
    max_rad_lin_area = 110
    for d, r in calibration_dict.items():
        if min_rad_lin_area <= r <= max_rad_lin_area:
            distances.append(d)
            radii.append(r)
    distances = np.array(distances)
    radii = np.array(radii)
    # fit_mask = (max_rad_lin_area >= radii) & (radii >= min_rad_lin_area)
    # fit_ds = distances[fit_mask]
    # fit_rs = radii[fit_mask]
    # lin_result = linregress(fit_rs, fit_ds)
    lin_result = linregress(radii, distances)
    slope = lin_result.slope
    intercept = lin_result.intercept
    return slope, intercept, min_rad_lin_area, max_rad_lin_area


if __name__ == "__main__":
    cal_json = "calibration_data.json"
    calibration_dict = load_calibration(cal_json)
    slope, intercept, min_rad_lin_area, max_rad_lin_area = get_calibration_function_and_limits(cal_json)
    all_distances, all_radii = [], []
    for d, r in calibration_dict.items():
        all_distances.append(d)
        all_radii.append(r)
    distances = np.array(all_distances)
    radii = np.array(all_radii)
    fit_rs = np.linspace(min_rad_lin_area, max_rad_lin_area, 200)
    fit_ds = slope * fit_rs + intercept
    plt.figure()
    plt.scatter(all_radii, all_distances)
    plt.plot(fit_rs, fit_ds, 'k--')
    plt.grid()
    plt.xlabel('radius (pixels)')
    plt.ylabel('distance ($\\mu$m)')
    plt.show()
