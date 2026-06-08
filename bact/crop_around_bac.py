import csv
import os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
MUM_IN_PIX = 0.225
SECS_PER_FRAME = 15.7 / 149


def crop_around(image, x, y, size=101):
    print(image.shape)
    is_cut = False
    half = size // 2
    # Calculate bounds
    x1 = max(0, x - half)
    x2 = min(image.shape[1], x + half + 1)
    y1 = max(0, y - half)
    y2 = min(image.shape[0], y + half + 1)
    if x1==0 or x2==image.shape[1] or y1==0 or y2==image.shape[0]:
        print(x,y)
        is_cut = True

    # Crop the region
    cropped = image[y1:y2, x1:x2]

    return cropped, is_cut


def pix2mum(pix):
    return pix * MUM_IN_PIX


def frame2time(frame):
    return frame * SECS_PER_FRAME


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


with open(r'up2\track_spots.csv', 'r') as f:
    tacks_no_per_bac_down = [[72, 348, 1461], [117, 268, 683, 1307, 1401], [942]]
    ignore_track_ids_down = [425, 588]
    ignore_track_ids_up = []
    keep_tracks_down = [1461, 1307, 268]
    keep_tracks_up = [37, 1418, 1627]
    preferred_up = [1627]
    tracks_down = [{}, {}, {}]
    tracks_up = [{}]
    tracks = tracks_up
    keep_tracks = keep_tracks_up
    preferred_keep = preferred_up
    ignore_track_ids = ignore_track_ids_up
    xs = []
    ys = []
    # new_1418_xs = []
    # new_1418_ys = []
    # new_1626_xs = []
    # new_1626_ys = []
    # new_1627_xs = []
    # new_1627_ys = []
    # newer_1627_xs = []
    # newer_1627_ys = []
    frames = []
    times = []
    reader = csv.DictReader(f)
    rows = [row for row in reader if row['TRACK_ID'].isdigit()]
    last_frame = -1
    last_tid = 0
    for row in rows:
        should_add = True
        tid = int(row['TRACK_ID'])
        # if tid in ignore_track_ids:
        #     continue
        frame = int(row['FRAME'])
        # bac_i = [i for i, tids in enumerate(tacks_no_per_bac_down) if tid in tids]
        # assert len(bac_i) == 1
        # bac_i = bac_i[0]
        # if len(bac_i) != 1:
        #     print(tid)
        #     print([i for i, tids in enumerate(tacks_no_per_bac)])
        #     print([tids for i, tids in enumerate(tacks_no_per_bac)])
        # bac_i = bac_i[0]
        bac_i = 0  # up
        if tid not in tracks[bac_i].keys():
            tracks[bac_i][tid] = [[float(row['POSITION_X'])], [float(row['POSITION_Y'])], [frame]]
        else:
            tracks[bac_i][tid][0].append(float(row['POSITION_X']))
            tracks[bac_i][tid][1].append(float(row['POSITION_Y']))
            tracks[bac_i][tid][2].append(frame)

single_bac_tracks = {}
for bac_i, bac_tracks in enumerate(tracks):
    single_track = None
    double_dots = []
    prev_double_dots = []
    double_frames = []
    for id, t in bac_tracks.items():
        if not single_track:  # first track to process
            single_track = [t[0], t[1], t[2]]
            continue
        pop_indices = []
        for idx, frame in enumerate(t[2]):
            if frame in single_track[2]:
                prev_idx = single_track[2].index(frame)
                # pop_indices.append(prev_idx)
                double_dots.append((t[0][idx], t[1][idx]))
                prev_double_dots.append((single_track[0][prev_idx], single_track[1][prev_idx]))
                double_frames.append((id, t[2][idx]))
                if id in keep_tracks:
                    # print(f'keeping track {id} dots')
                    pop_indices.append(prev_idx)
                else:
                    # print(f'throwing track {id} dots')
                    pop_indices.append(idx)
                pop_indices = sorted(pop_indices, reverse=True)
        if id in keep_tracks:
            for i in pop_indices:
                single_track[0].pop(i)
                single_track[1].pop(i)
                single_track[2].pop(i)
        else:
            for i in pop_indices:
                t[0].pop(i)
                t[1].pop(i)
                t[2].pop(i)
        single_track[0] += t[0]
        single_track[1] += t[1]
        single_track[2] += t[2]
    track_frames = np.array(single_track[2])
    sorted_indices = np.argsort(track_frames)
    sorted_frames = track_frames[sorted_indices]
    sorted_xs = np.array(single_track[0])[sorted_indices]
    sorted_ys = np.array(single_track[1])[sorted_indices]
    single_bac_tracks[bac_i] = [sorted_xs, sorted_ys, sorted_frames]

    plt.figure()
    plt.scatter(single_track[0], single_track[1], s=3)
    # for dot in double_dots:
    #     plt.scatter(dot[0], dot[1], c='r', s=3)
    # for dot in prev_double_dots:
    #     plt.scatter(dot[0], dot[1], c='g', s=3)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show(block=True)
#plt.show()
# for bac_tracks in tracks:
#     plt.figure()
#     for id, t in bac_tracks.items():
#         plt.scatter(t[0], t[1], s=3, label=f'id: {id}')
#     #plt.scatter(new_1418_xs, new_1418_ys, s=3, label='new 1418')
#     #plt.scatter(new_1626_xs, new_1626_ys, s=3, label='new 1626')
#     #plt.scatter(new_1627_xs, new_1627_ys, s=3, label='new 1627')
#     #plt.scatter(newer_1627_xs, newer_1627_ys, s=3, label='newer 1627')
#     plt.gca().invert_yaxis()
#     plt.axis('equal')
#     plt.legend()
#     plt.show(block=False)
# plt.show()
        # if frame == last_frame:
        #     if tid != last_tid:
        #         if tid == 651:
        #             xs.pop(-1)
        #             ys.pop(-1)
        #             frames.pop(-1)
        #         elif tid == 1418:
        #             new_1418_xs.append(float(row['POSITION_X']))
        #             new_1418_ys.append(float(row['POSITION_Y']))
        #             xs.pop(-1)
        #             ys.pop(-1)
        #             frames.pop(-1)
        #         elif tid == 1626:
        #             new_1626_xs.append(float(row['POSITION_X']))
        #             new_1626_ys.append(float(row['POSITION_Y']))
        #             should_add = False
        #         elif tid == 1627 and frame < 276:
        #             new_1627_xs.append(float(row['POSITION_X']))
        #             new_1627_ys.append(float(row['POSITION_Y']))
        #             xs.pop(-1)
        #             ys.pop(-1)
        #             frames.pop(-1)
        #         elif tid == 1627 and frame >= 276:
        #             newer_1627_xs.append(float(row['POSITION_X']))
        #             newer_1627_ys.append(float(row['POSITION_Y']))
        #             xs.pop(-1)
        #             ys.pop(-1)
        #             frames.pop(-1)
        #     else:
        #         print('wronggggggg')
        #         print(frame)
        #         print(last_frame)
        #         print(tid)
        # elif frame < last_frame:
        #     print('whattttt')
        # elif frame != last_frame + 1:
        #     missing_amount = frame - last_frame - 1
        #     for i in range(missing_amount):
        #         xs.append(0)
        #         ys.append(0)
        # if should_add:
        #     xs.append(float(row['POSITION_X']))
        #     ys.append(float(row['POSITION_Y']))
        #     frames.append(frame)
        # last_frame = frame
        # last_tid = tid
for bac_id, bac_t in single_bac_tracks.items():
    print(bac_id)
    if bac_id != 0:
        continue
    frames = bac_t[2]
    xs = bac_t[0]
    ys = bac_t[1]
    # print(frames)
    missing_frames = find_missing_frames_sorted(frames, 299)
    print(missing_frames)
    # if len(xs) != 300:
    #     print('oops')
    #     print(len(xs))
    # if len(ys) != 300:
    #     print('yoops')
    #     print(len(ys))
    # last_frame = -1
    # for frame in frames:
    #     if frame != last_frame + 1:
    #         print(frame, last_frame)
    #     last_frame = frame
    #
    images_dir = r'up2\Default'
    images_out_dir = fr'up2\Cropped_1\bac_id_{bac_id}'
    if not os.path.exists(images_out_dir):
        os.makedirs(images_out_dir)
    images = [os.path.join(images_dir, d) for d in os.listdir(images_dir) if d.endswith('tif')]
    for image in images:
        img = tiff.imread(image)
        image_name = os.path.split(image)[-1]
        frame = int(image_name.split('_')[-2][4:])
        if frame not in frames:
            print(f'skipping untracked frame no. {frame}')
            continue
        frame_idx = np.where(frames == frame)[0][0]
        x = int(round(xs[frame_idx]))
        y = int(round(ys[frame_idx]))
        if x == 0 or y == 0:
            print('Haaaa????')
        cropped, is_cutted = crop_around(img, x, y, 151)
        if is_cutted:
            print(f'Note! image of frame {frame} is cut for bac id {bac_id}')
        tiff.imwrite(os.path.join(images_out_dir, 'cropped_'+image_name), cropped)

    # plt.figure()
    # plt.scatter(xs, ys, s=3, label='all')
    # #plt.scatter(new_1418_xs, new_1418_ys, s=3, label='new 1418')
    # #plt.scatter(new_1626_xs, new_1626_ys, s=3, label='new 1626')
    # #plt.scatter(new_1627_xs, new_1627_ys, s=3, label='new 1627')
    # #plt.scatter(newer_1627_xs, newer_1627_ys, s=3, label='newer 1627')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()
