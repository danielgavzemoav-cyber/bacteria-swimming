import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def save_flat_npz(save_dir_path, bacteria_data):
    flat_dict = {
        f"{bid}_xs": data['xs']
        for bid, data in bacteria_data.items()
    }
    flat_dict.update({
        f"{bid}_ys": data['ys']
        for bid, data in bacteria_data.items()
    })
    flat_dict.update({
        f"{bid}_frames": data['frames']
        for bid, data in bacteria_data.items()
    })
    flat_dict.update({
        f"{bid}_missing": data['skipped_frames']
        for bid, data in bacteria_data.items()
    })
    save_path = os.path.join(save_dir_path, "bacteria_tracks.npz")
    np.savez_compressed(save_path, **flat_dict)


def validate_double_frames(particle_tracks):
    all_times = []
    doublings = []
    for trk in particle_tracks:
        curr_times = list(particle_tracks[trk]['times'])
        for t in curr_times:
            if t in all_times:
                doublings.append((t, trk))
        all_times += curr_times
    if len(doublings) != 0:
        for d in doublings:
            print(f'frame {d[0]} appears also in track id {d[1]}')
        return False
    else:
        return True


def find_missing_frames_sorted(sorted_frames, expected_last):
    missing_frames = []
    expected_frame = 0
    for fr in sorted_frames:
        if fr != expected_frame:
            diff = fr - expected_frame
            assert diff > 0
            for frame_i in range(expected_frame, expected_frame+diff):
                missing_frames.append(frame_i)
        expected_frame = fr + 1
    if sorted_frames[-1] < expected_last:
        for frame_i in range(sorted_frames[-1]+1, expected_last+1):
            missing_frames.append(frame_i)
    elif sorted_frames[-1] > expected_last:
        print(f'Warning! last frame found is later then the expected last frame!'
              f'expected: {expected_last}, found: {sorted_frames[-1]}')
    return missing_frames


def get_particle_idx_from_track_id(t_id, tracks_particles_mapping):
    found_ps = []
    for p in tracks_particles_mapping:
        if t_id in tracks_particles_mapping[p]:
            found_ps.append(p)
    assert len(found_ps) == 1
    return found_ps[0]


def stage_1_present_tracks_by_id(all_tracks):
    all_ids = list(all_tracks.keys())
    for i in range(math.ceil(len(all_tracks)/10)):
        start_idx = i * 10
        end_idx = (i + 1) * 10 if (i + 1) * 10 < len(all_ids) else len(all_ids)
        track_ids = all_ids[start_idx:end_idx]
        plt.figure()
        for track_id in track_ids:
            x_locs = all_tracks[track_id]['x_locations']
            y_locs = all_tracks[track_id]['y_locations']
            plt.scatter(x_locs, y_locs, label=f"track_id: {track_id}")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show(block=False)
    plt.figure()
    for track_id in all_tracks:
        x_locations = all_tracks[track_id]['x_locations']
        y_locations = all_tracks[track_id]['y_locations']
        plt.scatter(x_locations, y_locations, label=f"track_id: {track_id}")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()


def stage_2_validate_all_tracks_to_pids(all_tracks, tracks_per_particle):
    all_track_ids = list(all_tracks.keys())
    track_ids_from_particles = []
    for pid in tracks_per_particle:
        track_ids_from_particles += tracks_per_particle[pid]
    all_track_ids.sort()
    track_ids_from_particles.sort()
    if all_track_ids == track_ids_from_particles:
        print('Hurey!')
    else:
        print(all_track_ids)
        print(track_ids_from_particles)
        print('Boooo :(')
    assert all_track_ids == track_ids_from_particles


def stage_3_modify_spots_csv_remove_doubles(particle_tracks_tids):
    for pid in particle_tracks_tids:
        print(f'starting process pid: {pid}')
        p_tracks = particle_tracks_tids[pid]
        validate_double_frames(p_tracks)
        plt.figure()
        plt.title(f'particle index: {pid}')
        for track_id in p_tracks:
            x_locations = p_tracks[track_id]['x_locations']
            y_locations = p_tracks[track_id]['y_locations']
            plt.scatter(x_locations, y_locations, label=f"track_id: {track_id}")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show(block=False)
    plt.show()

# ------ data for down_0 directory ------- #
# current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_0'
# file_names = ['down_0_spots_1.csv', 'down_0_spots_2.csv']
# fpaths = [os.path.join(current_dir, file_name) for file_name in file_names]
# tracks_per_particle_1 = {6: [195, 1390], 7: [499, 4676]}
# tracks_per_particle_2 = {1: [339, 1729, 2877, 2313, 3626], 2: [188, 4034, 3634], 3: [222, 224, 2787],
#                          4: [3703, 2242, 2961, 3585], 5: [3542, 3543, 3913]}
# tracks_per_particle = {**tracks_per_particle_2, **tracks_per_particle_1}


# ------ data for down_1 directory ------- #
# current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_1'
# file_names = ['down_spots.csv']
# fpaths = [os.path.join(current_dir, file_name) for file_name in file_names]
# tracks_per_particle = {1: [318, 1382, 3196, 3447],
#                        2: [241, 955, 1122, 1391, 1749, 2347, 3071, 2443],
#                        3: [266, 684, 28, 336, 1183, 1255, 1996, 2277, 2437, 2479],
#                        4: [302, 725, 1477, 2193, 2211, 2462, 2588],
#                        5: [2749],
#                        6: [104]}
# excluded_tracks = []


# ------ data for down_2 directory ------- #
# current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_2'
# file_names = ['down_spots_2.csv', 'down_spots_3.csv']
# fpaths = [os.path.join(current_dir, file_name) for file_name in file_names]
# tracks_per_particle = {1: [92],
#                        2: [1698, 2353, 2568, 3857],
#                        3: [579, 3492],
#                        4: [195, 1717],
#                        5: [115],
#                        6: [22, 88],
#                        7: [509]}
# excluded_tracks = [384, 417, 138, 3032, 1534, 3447, 129, 491, 955, 1158, 3945]

# ------ data for down_3 directory ------- #
# current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_3'
# file_names = ['down_spots.csv']
# fpaths = [os.path.join(current_dir, file_name) for file_name in file_names]
# tracks_per_particle = {1: [587, 3073],
#                        2: [108, 1695, 2787, 1636],
#                        3: [760, 2053, 3334, 4286],
#                        4: [692, 1697],
#                        5: [604, 1873],
#                        6: [617],
#                        7: [87],
#                        8: [164, 2343],
#                        9: [78]}
# excluded_tracks = []


# ------ data for down_4 directory ------- #
current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_4'  # put your path here
file_names = ['down_spots.csv']
fpaths = [os.path.join(current_dir, file_name) for file_name in file_names]
tracks_per_particle = {1: [591, 1407, 1642, 3898],
                       2: [206, 1215],
                       3: [3],
                       4: [338, 236, 1376, 3718]}
excluded_tracks = []


# ------ data for down_5 directory ------- #
# current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_5'  # put your path here
# file_names = ['down_spots.csv']
# fpaths = [os.path.join(current_dir, file_name) for file_name in file_names]
# tracks_per_particle = {1: [267],
#                        2: [313],
#                        3: [31, 712, 983, 2107, 2146, 2249],
#                        4: [326, 2634],
#                        5: [1939],
#                        6: [247, 342, 690],
#                        7: [819, 2190, 2447, 2601]}
# excluded_tracks = []


all_tracks = {}  # {track_id: {'x_locations': [], 'y_locations': [], 'times': []}}
particle_tracks_tids = {} # {1: {tid: {'x_locations': [], 'y_locations': [], 'times': []}, tid: {...}}, 2: {tid: ...}}
particle_entire_tracks = {}

for fpath in fpaths:
    with open(fpath, 'r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row['TRACK_ID'].isdigit()]
        for row in rows:
            tid = int(row['TRACK_ID'])
            if tid in excluded_tracks:
                continue
            if tid not in all_tracks:
                all_tracks[tid] = {'x_locations': [], 'y_locations': [], 'times': []}
            all_tracks[tid]['x_locations'].append(float(row['POSITION_X']))
            all_tracks[tid]['y_locations'].append(float(row['POSITION_Y']))
            all_tracks[tid]['times'].append(int(row['FRAME']))

for track in all_tracks:
    all_tracks[track]['x_locations'] = np.array(all_tracks[track]['x_locations'])
    all_tracks[track]['y_locations'] = np.array(all_tracks[track]['y_locations'])
    all_tracks[track]['times'] = np.array(all_tracks[track]['times'])
    indices = np.argsort(all_tracks[track]['times'])
    x_locations = all_tracks[track]['x_locations'][indices]
    y_locations = all_tracks[track]['y_locations'][indices]
    times = all_tracks[track]['times'][indices]
    all_tracks[track]['x_locations'] = x_locations
    all_tracks[track]['y_locations'] = y_locations
    all_tracks[track]['times'] = times
    # pid = get_particle_idx_from_track_id(track, tracks_per_particle)
    # if pid not in particle_tracks_tids:
    #     particle_tracks_tids[pid] = {}
    # particle_tracks_tids[pid][track] = all_tracks[track]
# stage_1_present_tracks_by_id(all_tracks)
# stage_2_validate_all_tracks_to_pids(all_tracks, tracks_per_particle)
# code below is needed from stage 3 and on!
for track in all_tracks:
    pid = get_particle_idx_from_track_id(track, tracks_per_particle)
    if pid not in particle_tracks_tids:
        particle_tracks_tids[pid] = {}
    particle_tracks_tids[pid][track] = all_tracks[track]
# stage_3_modify_spots_csv_remove_doubles(particle_tracks_tids)

# this is stage 4 below (everything):
for pid in particle_tracks_tids:
    particle_entire_tracks[pid] = {}
    p_tracks = particle_tracks_tids[pid]
    validated_double_frames = validate_double_frames(p_tracks)
    assert validated_double_frames
    x_locations, y_locations, frames = np.empty(0), np.empty(0), np.empty(0, dtype=int)
    for track_id in p_tracks:
        x_locations = np.concatenate((x_locations, p_tracks[track_id]['x_locations']))
        y_locations = np.concatenate((y_locations, p_tracks[track_id]['y_locations']))
        frames = np.concatenate((frames, p_tracks[track_id]['times']))
    indices = np.argsort(frames)
    x_locations = x_locations[indices]
    y_locations = y_locations[indices]
    frames = frames[indices]
    skipped_frames = find_missing_frames_sorted(frames, 299)
    particle_entire_tracks[pid]['xs'] = x_locations
    particle_entire_tracks[pid]['ys'] = y_locations
    particle_entire_tracks[pid]['frames']= frames
    particle_entire_tracks[pid]['skipped_frames'] = np.array(skipped_frames)
    print(f'pid: {pid},\nmissing frames: {skipped_frames}\n')

# save_flat_npz(current_dir, particle_entire_tracks)
plt.figure()
for pid in particle_entire_tracks:
    # if track_id in tracks_per_particle[1]:
    #     continue
    x_locations = particle_entire_tracks[pid]['xs']
    # if len(x_locations) < 10:
    #     continue
    y_locations = particle_entire_tracks[pid]['ys']
    plt.scatter(x_locations, y_locations, label=f"particle_id: {pid}")
plt.legend()
plt.gca().invert_yaxis()
plt.show()
