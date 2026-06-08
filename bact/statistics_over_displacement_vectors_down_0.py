import json
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_time_diffs(measure_metadata_file, missing_frames=None):
    with open(measure_metadata_file, "r") as f:
        metadata = json.load(f)
    # elapsed_times = []
    elapsed_times = [float(metadata[k]['ElapsedTime-ms']) for k in metadata.keys() if k.startswith('Metadata-Default')]
    elapsed_times.sort()
    elapsed_times = np.array(elapsed_times)  # just in case it is not according to the sequence
    dt_array = np.diff(elapsed_times)
    if missing_frames:
        frames_to_delete = []
        for fr in missing_frames:
            if (fr - 1) not in missing_frames:
                frames_to_delete.append(fr - 1)
            frames_to_delete.append(fr)
        dt_array = np.delete(dt_array, frames_to_delete)
    return dt_array


def get_u_v_frames_to_remove(missing_frames):
    missing_frames.sort()
    frames_to_remove = []
    removed = 0
    prev_removed = max(missing_frames) + 1
    for f in missing_frames:
        if f != prev_removed + 1:
            frames_to_remove.append(f - 1 - removed)
        prev_removed = f
        removed += 1
    return frames_to_remove


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


def load_vector_fields_matlab(file_path, smoothed=True, initial_index=0, final_index=None, missing_frames=None):
    mat = loadmat(file_path)

    u_key = 'u_smoothed' if smoothed else 'u_original'
    v_key = 'v_smoothed' if smoothed else 'v_original'

    u_raw = [np.array(f) for f in mat[u_key].squeeze()]
    v_raw = [np.array(f) for f in mat[v_key].squeeze()]
    x_raw = [np.array(f) for f in mat['x'].squeeze()]
    y_raw = [np.array(f) for f in mat['y'].squeeze()]

    if final_index is None:
        final_index = len(u_raw) - 1

    u_raw = u_raw[initial_index:final_index + 1]
    v_raw = v_raw[initial_index:final_index + 1]
    x_raw = x_raw[initial_index:final_index + 1]
    y_raw = y_raw[initial_index:final_index + 1]

    u_all = np.stack(u_raw, axis=2) * 1e6
    v_all = np.stack(v_raw, axis=2) * 1e6
    x_all = np.stack(x_raw, axis=2) * 1e6
    y_all = np.stack(y_raw, axis=2) * 1e6

    if missing_frames:
        frames_to_remove = get_u_v_frames_to_remove(missing_frames)
        u_all = np.delete(u_all, frames_to_remove, axis=2)
        v_all = np.delete(v_all, frames_to_remove, axis=2)
        x_all = np.delete(x_all, frames_to_remove, axis=2)
        y_all = np.delete(y_all, frames_to_remove, axis=2)

    return x_all, y_all, u_all, v_all


def calc_relative_and_rotation(x_all, y_all, u_all, v_all, frames_num):
    u_rotated = np.empty_like(u_all)
    v_rotated = np.empty_like(v_all)
    x_rotated = np.empty_like(x_all)
    y_rotated = np.empty_like(y_all)
    u_relative = np.empty_like(u_all)
    v_relative = np.empty_like(v_all)
    mean_velocities_rotated = []
    mean_velocities = []
    skipped_frames = []

    for i in range(frames_num):
        u_vecs = u_all[:, :, i]
        v_vecs = v_all[:, :, i]
        x = x_all[:, :, i]
        y = y_all[:, :, i]
        u_nan_ratio = np.isnan(u_all[:, :, i]).sum() / u_all[:, :, i].size
        v_nan_ratio = np.isnan(v_all[:, :, i]).sum() / v_all[:, :, i].size
        # if u_nan_ratio >= 3 or v_nan_ratio >= 3:
        #     print(f"skipping displacement frame {i} due to too many nans in hte displacement frame."
        #           f"u nan ratio is {u_nan_ratio:.2%}, v nan ratio is {v_nan_ratio:.2%}")
        #     skipped_frames.append(i)
        #     continue
        mean_u = np.nanmean(u_vecs)
        mean_v = np.nanmean(v_vecs)
        mean_velocities.append((mean_u, mean_v))
        theta = np.arctan2(mean_u, mean_v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta, cos_theta]])
        u_flat = u_vecs.flatten()
        v_flat = v_vecs.flatten()
        x_flat = x.flatten()
        y_flat = y.flatten()

        mean_vec = np.array([mean_u, mean_v])  # Shape: (2,)
        mean_rotated_vec = rot_matrix @ mean_vec  # Matrix multiplication
        mean_velocities_rotated.append(mean_rotated_vec)

        vecs = np.stack((u_flat, v_flat), axis=0)  # Shape: (2, N)
        rotated_vecs = rot_matrix @ vecs  # Rotate
        u_rotated[:, :, i] = rotated_vecs[0].reshape(u_vecs.shape)
        v_rotated[:, :, i] = rotated_vecs[1].reshape(v_vecs.shape)
        u_relative[:, :, i] = u_rotated[:, :, i] - mean_rotated_vec[0]
        v_relative[:, :, i] = v_rotated[:, :, i] - mean_rotated_vec[1]

        coords = np.stack((x_flat, y_flat), axis=0)  # Shape: (2, N)
        rotated_coords = rot_matrix @ coords
        x_rotated[:, :, i] = rotated_coords[0].reshape(x.shape)
        y_rotated[:, :, i] = rotated_coords[1].reshape(y.shape)

    return mean_velocities, mean_velocities_rotated, x_rotated, y_rotated, u_relative, v_relative


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
    # plt.quiver(
    #     x_center,
    #     y_center,
    #     mean_u_rot,
    #     mean_v_rot,
    #     scale=0.3,
    #     angles='xy',
    #     scale_units='xy',
    #     color='red',
    #     linewidth=2,
    #     label='Mean Velocity'
    # )
    plt.scatter(x_center, y_center, c='r', s=2)
    plt.gca().invert_yaxis()  # Optional: flip Y axis to match image conventions
    plt.title(f"Mean Relative Velocity Field")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show(block=False)


def visualize_frame_relative_bac(iframes, x_all, x_rot, y_all, y_rot, u_rel, v_rel, mean_rot):
    for iframe in iframes:
        mean_u_rot = mean_rot[iframe][0]
        mean_v_rot = mean_rot[iframe][1]

        # plt.figure(figsize=(6, 6))
        # plt.quiver(
        #     x_all[:, :, iframe],
        #     y_all[:, :, iframe],
        #     u_rel[:, :, iframe] + mean_u_rot,
        #     v_rel[:, :, iframe] + mean_v_rot,
        #     scale=10,  # Adjust scale for better appearance
        #     angles='xy',
        #     scale_units='xy',
        #     color='blue',
        #     label='Rotated Vectors'
        # )
        # x_center = np.mean(x_all[:, :, iframe])
        # y_center = np.mean(y_all[:, :, iframe])
        # plt.quiver(
        #     x_center,
        #     y_center,
        #     # mean_velocities[frame_index][0],
        #     # mean_velocities[frame_index][1],
        #     mean_u_rot,
        #     mean_v_rot,
        #     scale=10,
        #     angles='xy',
        #     scale_units='xy',
        #     color='red',
        #     linewidth=2,
        #     label='Mean Velocity'
        # )
        # plt.gca().invert_yaxis()  # Optional: flip Y axis to match image conventions
        # plt.title(f"Rotated Velocity Field, Un-Rotated Frame, Frame No. {iframe}")
        # plt.xlabel("x (μm)")
        # plt.ylabel("y (μm)")
        # plt.axis("equal")
        # plt.legend()
        # plt.grid(True)
        # plt.show(block=False)

        plt.figure(figsize=(6, 6))
        plt.quiver(
            x_all[:, :, iframe],
            y_all[:, :, iframe],
            u_rel[:, :, iframe],
            v_rel[:, :, iframe],
            scale=10,  # Adjust scale for better appearance
            angles='xy',
            scale_units='xy',
            color='blue',
            label='Relative Vectors'
        )
        x_center = np.mean(x_all[:, :, iframe])
        y_center = np.mean(y_all[:, :, iframe])
        plt.quiver(
            x_center,
            y_center,
            mean_u_rot,
            mean_v_rot,
            scale=10,
            angles='xy',
            scale_units='xy',
            color='red',
            linewidth=2,
            label='Mean Velocity'
        )
        plt.gca().invert_yaxis()  # Optional: flip Y axis to match image conventions
        plt.title(f"Relative Velocity Field (Rotated Frame) Frame No. {iframe}")
        plt.xlabel("x (μm)")
        plt.ylabel("y (μm)")
        plt.axis("equal")
        plt.grid(True)
        plt.show(block=False)


measure_dir = r'up2\Default'
vectors_dir = r'up2'
output_dir = r'up2'
frames_missed = [39]  # zero-based
measure_file = os.path.join(measure_dir, 'metadata.txt')
vectors_file = os.path.join(vectors_dir, 'PIVlab_5_151_crop.mat')
output_file = os.path.join(output_dir, 'velocity_data_5_151_crop.npz')
dts = get_time_diffs(measure_file, frames_missed)
xs, ys, us, vs = load_vector_fields_matlab(vectors_file, smoothed=False, missing_frames=frames_missed)
dt_reshaped = dts.reshape((1, 1, -1))
u_velocity = us / dt_reshaped
v_velocity = vs / dt_reshaped
num_frames = u_velocity.shape[2]
#print(num_frames)

are_locations_equal = verify_same_vectors_locations(xs, ys, num_frames)
assert are_locations_equal

u_nans_by_frame = {}
v_nans_by_frame = {}
all_u_nans_frames = []
all_v_nans_frames = []
for frame in range(num_frames):
    u_nan_indices = np.argwhere(np.isnan(u_velocity[:, :, frame]))
    v_nan_indices = np.argwhere(np.isnan(v_velocity[:, :, frame]))
    if u_nan_indices.size != 0:
        u_nans_by_frame[frame] = u_nan_indices
        if u_nan_indices.size == u_velocity[:, :, frame].size:
            all_u_nans_frames.append(frame)
    if v_nan_indices.size != 0:
        v_nans_by_frame[frame] = v_nan_indices
        if v_nan_indices.size == v_velocity[:, :, frame].size:
            all_v_nans_frames.append(frame)
print(list(u_nans_by_frame.keys()))
print(all_u_nans_frames)
print([val.size for val in u_nans_by_frame.values()])
print(list(v_nans_by_frame.keys()))
print(all_v_nans_frames)
print([val.size for val in v_nans_by_frame.values()])

mean_velocities, mean_velocities_rotated, x_rotated, y_rotated, u_relative, v_relative = calc_relative_and_rotation(
    xs, ys, u_velocity, v_velocity, num_frames
)
mean_velocities_rotated = np.array(mean_velocities_rotated)

magnitudes = np.linalg.norm(mean_velocities_rotated, axis=1)
mean_magnitude = np.mean(magnitudes)
mean_vel = (0, mean_magnitude)

u_rel_time_mean, v_rel_time_mean = mean_over_time(u_relative, v_relative)
visualize_velocity_field(xs, ys, u_rel_time_mean, v_rel_time_mean, mean_vel)
np.savez_compressed(
    output_file,
    x_all=xs,
    y_all=ys,
    u_rel=u_relative,
    v_rel=v_relative,
    mean_velocities=mean_velocities,
    mean_velocities_rotated=mean_velocities_rotated
)

#visualize_frame_relative_bac([0,2,4], xs, x_rotated, ys, y_rotated, u_relative, v_relative, mean_velocities_rotated)
plt.show(block=True)
