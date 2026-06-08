import json
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path

from crop_around_after_analysis import load_bac_tracks_from_npz

MUM_IN_PIX = 0.225


def get_bac_velocities_from_track(bac_track, dts, missing_frames=None):
    uncropped_frames = bac_track['frames']
    track_xs_pxl = bac_track['xs']
    track_ys_pxl = bac_track['ys']
    missing_frames_npz = bac_track['missing']
    frames_to_delete = []
    # if missing_frames:
    #     assert all(mfnpz in missing_frames for mfnpz in missing_frames_npz)
    # if len(missing_frames_npz) != 0:
    #     assert missing_frames
    if missing_frames:
        full_length = len(uncropped_frames) + len(missing_frames_npz)
        xs_padded = np.zeros(full_length, dtype=xs.dtype)
        ys_padded = np.zeros(full_length, dtype=ys.dtype)
        xs_padded[uncropped_frames] = track_xs_pxl
        ys_padded[uncropped_frames] = track_ys_pxl
        padded_xs_diffs = np.diff(xs_padded) * MUM_IN_PIX
        padded_ys_diffs = np.diff(ys_padded) * MUM_IN_PIX
        for fr in missing_frames:
            if fr != 0 and (fr - 1) not in missing_frames:
                frames_to_delete.append(fr - 1)
            if fr != padded_xs_diffs.size:
                frames_to_delete.append(fr)
        bac_dxs = np.delete(padded_xs_diffs, frames_to_delete)
        bac_dys = np.delete(padded_ys_diffs, frames_to_delete)
    else:
        bac_dxs = np.diff(track_xs_pxl) * MUM_IN_PIX
        bac_dys = np.diff(track_ys_pxl) * MUM_IN_PIX
    bac_us = bac_dxs / dts
    bac_vs = bac_dys / dts
    return bac_us, bac_vs


def get_time_diffs(measure_metadata_file, missing_frames=None):
    with open(measure_metadata_file, "r") as f:
        metadata = json.load(f)
    # elapsed_times = []
    elapsed_times = [float(metadata[k]['ElapsedTime-ms']) for k in metadata.keys() if k.startswith('Metadata-Default')]
    elapsed_times.sort()
    elapsed_times = np.array(elapsed_times)  # just in case it is not according to the sequence
    dt_array = np.diff(elapsed_times)
    print(missing_frames)
    if missing_frames:
        frames_to_delete = []
        for fr in missing_frames:
            if fr != 0 and (fr - 1) not in missing_frames:
                frames_to_delete.append(fr - 1)
            if fr != dt_array.size:
                frames_to_delete.append(fr)
        print(frames_to_delete)
        dt_array = np.delete(dt_array, frames_to_delete)
    return dt_array


def get_u_v_frames_to_remove(missing_frames, last_possible_frame=299):
    missing_frames.sort()
    frames_to_remove = []
    removed = 0
    prev_removed = max(missing_frames) + 1
    # if all last frames are missing, no need to remove a u/v frame for them cause there is no such
    if last_possible_frame in missing_frames:
        diffs = np.diff(missing_frames)
        idx = len(diffs)
        while idx > 0 and diffs[idx - 1] == 1:
            idx -= 1
        missing_frames = missing_frames[:idx]
    for f in missing_frames:
        if f != prev_removed + 1 and f != 0:
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

    u_all = np.stack(u_raw, axis=2) * 1e3
    v_all = np.stack(v_raw, axis=2) * 1e3
    x_all = np.stack(x_raw, axis=2) * 1e6
    y_all = np.stack(y_raw, axis=2) * 1e6

    if missing_frames:
        frames_to_remove = get_u_v_frames_to_remove(missing_frames)
        u_all = np.delete(u_all, frames_to_remove, axis=2)
        v_all = np.delete(v_all, frames_to_remove, axis=2)
        x_all = np.delete(x_all, frames_to_remove, axis=2)
        y_all = np.delete(y_all, frames_to_remove, axis=2)

    return x_all, y_all, u_all, v_all


def calc_relative_and_rotation(x_all, y_all, u_all, v_all, bac_us, bac_vs, frames_num):
    u_rotated = np.empty_like(u_all)
    v_rotated = np.empty_like(v_all)
    u_rotated2 = np.empty_like(u_all)
    v_rotated2 = np.empty_like(v_all)
    x_rotated = np.empty_like(x_all)
    y_rotated = np.empty_like(y_all)
    u_relative = np.empty_like(u_all)
    v_relative = np.empty_like(v_all)
    u_relative2 = np.empty_like(u_all)
    v_relative2 = np.empty_like(v_all)
    u_rel_rotated = np.empty_like(u_all)
    v_rel_rotated = np.empty_like(v_all)
    mean_velocities_rotated = []
    mean_velocities_rotated2 = []
    mean_velocities = []
    skipped_frames = []
    assert frames_num == len(bac_us)
    for i in range(frames_num):
        u_vecs = u_all[:, :, i]
        v_vecs = v_all[:, :, i]
        x = x_all[:, :, i]
        y = y_all[:, :, i]
        bac_u = bac_us[i]
        bac_v = bac_vs[i]
        u_nan_ratio = np.isnan(u_all[:, :, i]).sum() / u_all[:, :, i].size
        v_nan_ratio = np.isnan(v_all[:, :, i]).sum() / v_all[:, :, i].size
        # if u_nan_ratio >= 3 or v_nan_ratio >= 3:
        #     print(f"skipping displacement frame {i} due to too many nans in hte displacement frame."
        #           f"u nan ratio is {u_nan_ratio:.2%}, v nan ratio is {v_nan_ratio:.2%}")
        #     skipped_frames.append(i)
        #     continue
        mean_u = np.nanmean(u_vecs)
        mean_v = np.nanmean(v_vecs)
        print(f'bac u from npz:{-bac_u}, bac u from mean u filed: {mean_u}')
        print(f'bac v from npz:{-bac_v}, bac v from mean v filed: {mean_v}')
        mean_velocities.append((mean_u, mean_v))
        theta = np.arctan2(-bac_u, -bac_v)
        theta2 = np.arctan2(mean_u, mean_v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_theta2 = np.cos(theta2)
        sin_theta2 = np.sin(theta2)
        rot_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta, cos_theta]])
        rot_matrix2 = np.array([[cos_theta2, -sin_theta2],
                               [sin_theta2, cos_theta2]])

        # started change here #
        u_substracted_vecs = u_vecs + bac_u
        v_substracted_vecs = v_vecs + bac_v
        u_rel_flat = u_substracted_vecs.flatten()
        v_rel_flat = v_substracted_vecs.flatten()
        rel_vecs = np.stack((u_rel_flat, v_rel_flat), axis=0)  # Shape: (2, N)
        rotated_rel_vecs = rot_matrix @ rel_vecs
        u_rel_rotated[:, :, i] = rotated_rel_vecs[0].reshape(u_vecs.shape)
        v_rel_rotated[:, :, i] = rotated_rel_vecs[1].reshape(v_vecs.shape)
        # u_substracted_vecs = u_vecs - mean_u
        # v_substracted_vecs = v_vecs - mean_v
        # u_rel_flat = u_substracted_vecs.flatten()
        # v_rel_flat = v_substracted_vecs.flatten()
        # rel_vecs = np.stack((u_rel_flat, v_rel_flat), axis=0)  # Shape: (2, N)
        # rotated_rel_vecs = rot_matrix2 @ rel_vecs
        # u_rel_rotated[:, :, i] = rotated_rel_vecs[0].reshape(u_vecs.shape)
        # v_rel_rotated[:, :, i] = rotated_rel_vecs[1].reshape(v_vecs.shape)
        # ---- #
        u_flat = u_vecs.flatten()
        v_flat = v_vecs.flatten()
        x_flat = x.flatten()
        y_flat = y.flatten()
        mean_vec = np.array([mean_u, mean_v])  # Shape: (2,)
        mean_rotated_vec = rot_matrix2 @ mean_vec  # Matrix multiplication
        mean_velocities_rotated2.append(mean_rotated_vec)
        bac_velocity_vec_opp = np.array([-bac_u, -bac_v])
        rotated_bac_opp_vel_vec = rot_matrix @ bac_velocity_vec_opp
        mean_velocities_rotated.append(rotated_bac_opp_vel_vec)
        vecs = np.stack((u_flat, v_flat), axis=0)  # Shape: (2, N)
        rotated_vecs = rot_matrix @ vecs
        rotated_vecs2 = rot_matrix2 @ vecs# Rotate
        u_rotated[:, :, i] = rotated_vecs[0].reshape(u_vecs.shape)
        v_rotated[:, :, i] = rotated_vecs[1].reshape(v_vecs.shape)
        u_rotated2[:, :, i] = rotated_vecs2[0].reshape(u_vecs.shape)
        v_rotated2[:, :, i] = rotated_vecs2[1].reshape(v_vecs.shape)
        u_relative2[:, :, i] = u_rotated2[:, :, i] - mean_rotated_vec[0]
        v_relative2[:, :, i] = v_rotated2[:, :, i] - mean_rotated_vec[1]
        u_relative[:, :, i] = u_rotated[:, :, i] - rotated_bac_opp_vel_vec[0]
        v_relative[:, :, i] = v_rotated[:, :, i] - rotated_bac_opp_vel_vec[1]
        coords = np.stack((x_flat, y_flat), axis=0)  # Shape: (2, N)
        rotated_coords = rot_matrix @ coords
        x_rotated[:, :, i] = rotated_coords[0].reshape(x.shape)
        y_rotated[:, :, i] = rotated_coords[1].reshape(y.shape)

    # return mean_velocities, mean_velocities_rotated, x_rotated, y_rotated, u_relative, v_relative
    return mean_velocities, mean_velocities_rotated, x_rotated, y_rotated, u_rel_rotated, v_rel_rotated


def mean_over_time(u_rel, v_rel):
    mean_u_rel = np.nanmean(u_rel, axis=2)  # shape: (H, W)
    mean_v_rel = np.nanmean(v_rel, axis=2)  # shape: (H, W)
    return mean_u_rel, mean_v_rel


def visualize_velocity_field(x_arr, y_arr, u_rel_mean, v_rel_mean, mean_bac_velocity, processed_dir, centered=False, should_center=True, ylims=None):
    plt.figure(figsize=(6, 6))
    x_center = np.mean(x_arr)
    y_center = np.mean(y_arr)
    if not centered and should_center:
        x_arr = x_arr - x_center
        y_arr = y_arr - y_center
    mean_bac_u = mean_bac_velocity[0]
    mean_bac_v = mean_bac_velocity[1]
    plt.quiver(
        x_arr,
        -y_arr,
        u_rel_mean,
        -v_rel_mean,
        scale=0.001,  # Adjust scale for better appearance
        angles='xy',
        scale_units='xy',
        color='blue',
        label='Velocity Field',
        zorder=3
    )
    x_center = 0 if should_center and not centered else x_center
    y_center = 0 if should_center and not centered else y_center
    plt.quiver(
        x_center,
        -y_center,
        mean_bac_u,
        -mean_bac_v,
        scale=0.001,
        angles='xy',
        scale_units='xy',
        color='red',
        linewidth=2,
        label='Bacteria Velocity',
        zorder=1,
        alpha=0.3
    )
    plt.scatter(x_center, y_center, c='r', s=2)
    # plt.gca().invert_yaxis()  # Optional: flip Y axis to match image conventions
    if processed_dir == 'all_together':
        plt.title(f"Mean Velocity Field")
    else:
        plt.title(f"Mean Velocity Field of dir: {processed_dir}")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    # if ylims:
    plt.ylim(-35, 35)
    # plt.axis("equal")
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


def find_missing_frames_sorted(sorted_frames, expected_last):
    missing_frms = []
    expected_frame = 0
    for fr in sorted_frames:
        if fr != expected_frame:
            diff = fr - expected_frame
            assert diff > 0
            for frame_i in range(expected_frame, expected_frame+diff):
                missing_frms.append(frame_i)
        expected_frame = fr + 1
    if sorted_frames[-1] != expected_last:
        for frame_i in range(sorted_frames[-1]+1, expected_last+1):
            missing_frms.append(frame_i)
    return missing_frms


grandmother_dir = r'C:\Users\shsch\OneDrive\Desktop\5'
tracks_npz_name = r'bacteria_tracks.npz'
mother_dirs = [str(p) for p in Path(grandmother_dir).iterdir() if p.is_dir() and ('bacterias_per_z' not in p.name)]
combined_u_rels, combined_v_rels, combined_magnitudes, combined_xs, combined_ys = None, None, None, None, None
for mother_dir in mother_dirs:
    measure_dir = os.path.join(mother_dir, 'Default')
    measure_file = os.path.join(measure_dir, 'metadata.txt')
    single_bac_tracks = load_bac_tracks_from_npz(os.path.join(mother_dir, tracks_npz_name))
    excluded_bac_ids = ['2', '5'] if 'down_0' in mother_dir else []

    cropped_imgs_dirs = [str(p) for p in Path(os.path.join(mother_dir, 'Cropped')).iterdir()
                         if p.is_dir() and not any(x in p.name for x in excluded_bac_ids)]

    for bac_images_dir in cropped_imgs_dirs:
        currently_processed = Path(bac_images_dir).parts[-3:]
        currently_processed = os.path.join(*currently_processed)
        bac_id = currently_processed[-1]
        # currently_processed = os.path.join(*(os.path.split(bac_images_dir)[-4:]))
        print(f'processing directory: {currently_processed}')
        frames = [int(d.split('_')[-2][4:]) for d in os.listdir(bac_images_dir) if d.endswith('tif')]
        frames_missed = find_missing_frames_sorted(frames, 299)  # zero-based

        vectors_file = os.path.join(bac_images_dir, 'PIVlab.mat')
        dts = get_time_diffs(measure_file, frames_missed)
        xs, ys, us, vs = load_vector_fields_matlab(vectors_file, smoothed=False, missing_frames=frames_missed)
        dt_reshaped = dts.reshape((1, 1, -1))
        u_velocity = us / dt_reshaped
        v_velocity = vs / dt_reshaped
        bac_us, bac_vs = get_bac_velocities_from_track(single_bac_tracks[bac_id], dts, frames_missed)
        num_frames = u_velocity.shape[2]
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
        print('frames with all us nans: ', all_u_nans_frames)
        print('frames with u nans: ', list(u_nans_by_frame.keys()))
        print('amount of nans in each frame: ', [val.size for val in u_nans_by_frame.values()])
        print('frames with all vs nans: ', all_v_nans_frames)
        print('frames with v nans: ', list(v_nans_by_frame.keys()))
        print('amount of nans in each frame: ', [val.size for val in v_nans_by_frame.values()])

        mean_velocities, mean_velocities_rotated, x_rotated, y_rotated, u_relative, v_relative = calc_relative_and_rotation(
            xs, ys, u_velocity, v_velocity, bac_us, bac_vs, num_frames
        )
        mean_velocities_rotated = np.array(mean_velocities_rotated)

        magnitudes = np.linalg.norm(mean_velocities_rotated, axis=1)
        mean_magnitude = np.mean(magnitudes)
        mean_vel = (0, mean_magnitude)
        bac_mean_vel = (-mean_vel[0], -mean_vel[1])
        if combined_u_rels is None:
            combined_magnitudes = magnitudes
            combined_u_rels = u_relative
            combined_v_rels = v_relative
            combined_xs = xs
            combined_ys = ys
        else:
            combined_u_rels = np.concatenate((combined_u_rels, u_relative), axis=2)
            combined_v_rels = np.concatenate((combined_v_rels, v_relative), axis=2)
            combined_magnitudes = np.concatenate((combined_magnitudes, magnitudes))
            combined_xs = np.concatenate((combined_xs, xs), axis=2)
            combined_ys = np.concatenate((combined_ys, ys), axis=2)
        u_rel_time_mean, v_rel_time_mean = mean_over_time(u_relative, v_relative)
        #visualize_velocity_field(xs[:, :, 0], ys[:, :, 0], u_rel_time_mean, v_rel_time_mean, bac_mean_vel, currently_processed)

num_frames = combined_u_rels.shape[2]
are_locations_equal = verify_same_vectors_locations(combined_xs, combined_ys, num_frames)
assert are_locations_equal
xs = combined_xs[:, :, 0]
ys = combined_ys[:, :, 0]
x_center = np.mean(xs)
y_center = np.mean(ys)
x_centered = xs - x_center
y_centered = ys - y_center
mean_magnitude = np.mean(combined_magnitudes)
mean_vel = (0, mean_magnitude)
bac_mean_vel = (-mean_vel[0], -mean_vel[1])
u_rel_time_mean, v_rel_time_mean = mean_over_time(combined_u_rels, combined_v_rels)
u_means_mean_along_y = np.mean(u_rel_time_mean, axis=1)
v_means_mean_along_y = np.mean(v_rel_time_mean, axis=1)
u_means_mean_along_y_cut = np.mean(u_rel_time_mean[:, 2:-2], axis=1)
v_means_mean_along_y_cut = np.mean(v_rel_time_mean[:, 2:-2], axis=1)
only_xs = x_centered[0, :]
visualize_velocity_field(x_centered, y_centered, u_rel_time_mean, v_rel_time_mean, bac_mean_vel, 'all_together', centered=True, ylims=(-30, 35))
visualize_velocity_field(only_xs, np.zeros_like(only_xs), u_means_mean_along_y, v_means_mean_along_y, bac_mean_vel, 'mean', centered=True)
visualize_velocity_field(only_xs, np.zeros_like(only_xs), u_means_mean_along_y_cut, v_means_mean_along_y_cut, bac_mean_vel, 'mean', centered=True)
u_rel_time_mean_no_center = u_rel_time_mean
u_rel_time_mean_no_center[7:10, 7:10] = np.nan
v_rel_time_mean_no_center = v_rel_time_mean
v_rel_time_mean_no_center[7:10, 7:10] = np.nan
visualize_velocity_field(x_centered, y_centered, u_rel_time_mean_no_center, v_rel_time_mean_no_center, bac_mean_vel, 'all_together', centered=True, ylims=(-30, 35))
plt.figure(figsize=(6, 2))
plt.bar(only_xs, u_means_mean_along_y * 1000)
plt.title("horizontal mean velocity vs horizontal location")
plt.xlabel("x (μm)")
plt.ylabel("horizontal velocity (μm/s)")
plt.ylim(max(v_means_mean_along_y)*1000/(8/3), -max(v_means_mean_along_y)*1000/(8/3))
plt.show(block=False)
plt.figure()
plt.bar(only_xs, -v_means_mean_along_y * 1000, zorder=3)
plt.title("vertical mean velocity vs horizontal location")
plt.xlabel("x (μm)")
plt.ylabel("vertical velocity (μm/s)")
plt.show(block=False)
plt.figure()
plt.bar(only_xs, -v_means_mean_along_y * 1000, zorder=3)
plt.bar(0, np.linalg.norm(bac_mean_vel)*1000, color='red', alpha=0.4, zorder=1)
plt.title("vertical mean velocity vs horizontal location")
plt.xlabel("x (μm)")
plt.ylabel("vertical velocity (μm/s)")
plt.show(block=False)
#  output_file = os.path.join(output_dir, 'velocity_data.npz')
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# np.savez_compressed(
#     output_file,
#     x_all=xs,
#     y_all=ys,
#     u_rel=u_relative,
#     v_rel=v_relative,
#     mean_velocities=mean_velocities,
#     mean_velocities_rotated=mean_velocities_rotated
# )

#visualize_frame_relative_bac([0,2,4], xs, x_rotated, ys, y_rotated, u_relative, v_relative, mean_velocities_rotated)
plt.show(block=True)
