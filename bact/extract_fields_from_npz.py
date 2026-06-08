import json
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def verify_same_vectors_locations(x_all, y_all, frames_amount):
    xeq, yeq = True, True
    for nf in range(1, frames_amount):
        x_equal = np.allclose(x_all[:, :, 0], x_all[:, :, nf])
        if not x_equal:
            xeq = False
            print(f'x not equal in frame {nf}')
        y_equal = np.allclose(y_all[:, :, 0], y_all[:, :, nf])
        if not y_equal:
            yeq = False
            print(f'y not equal in frame {nf}')
    is_eq = yeq and xeq
    return is_eq


def mean_over_time(u_rel, v_rel):
    mean_u_rel = np.nanmean(u_rel, axis=2)  # shape: (H, W)
    mean_v_rel = np.nanmean(v_rel, axis=2)  # shape: (H, W)
    return mean_u_rel, mean_v_rel


def visualize_velocity_field(x_all, y_all, u_rel_mean, v_rel_mean, mean_velocity):
    plt.figure(figsize=(6, 6))
    x_arr = x_all[:, :, 0]
    y_arr = y_all[:, :, 0]
    mean_u_rot = mean_velocity[0]
    mean_v_rot = mean_velocity[1]
    plt.quiver(
        x_arr,
        y_arr,
        u_rel_mean,
        v_rel_mean,
        scale=0.3,  # Adjust scale for better appearance
        angles='xy',
        scale_units='xy',
        color='blue',
        label='Mean Relative Velocity Field'
    )
    x_center = np.mean(x_arr)
    y_center = np.mean(y_arr)
    plt.quiver(
        x_center,
        y_center,
        mean_u_rot,
        mean_v_rot,
        scale=1,
        angles='xy',
        scale_units='xy',
        color='red',
        linewidth=2,
        label='Mean Velocity'
    )
    #plt.scatter(x_center, y_center, c='r', s=2)
    plt.gca().invert_yaxis()  # Optional: flip Y axis to match image conventions
    plt.title(f"Mean Relative Velocity Field")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show(block=False)


if __name__ == '__main__':
    measure_dir = r'up2\Default'
    npz_dir = r'up2'
    npz_file = os.path.join(npz_dir, 'velocity_data_1.npz')

    data = np.load(npz_file)
    xs = data['x_all']
    ys = data['y_all']
    u_relative = data['u_rel']
    v_relative = data['v_rel']
    mean_velocities = data['mean_velocities']
    mean_velocities_rotated = data['mean_velocities_rotated']
    u_rel_time_mean, v_rel_time_mean = mean_over_time(u_relative, v_relative)
    magnitudes = np.linalg.norm(mean_velocities_rotated, axis=1)
    mean_magnitude = np.mean(magnitudes)
    mean_vel = (0, mean_magnitude)
    visualize_velocity_field(xs, ys, u_rel_time_mean, v_rel_time_mean, mean_vel)

    plt.show(block=True)
