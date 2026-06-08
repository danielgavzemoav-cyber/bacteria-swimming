import csv
import os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
MUM_IN_PIX = 0.225
SECS_PER_FRAME = 15.7 / 149


def load_bac_tracks_from_npz(npz_path):
    loaded = np.load(npz_path)
    bacteria_data = {}
    for key in loaded.files:
        bid, data_name = key.split("_", 1)
        if bid not in bacteria_data:
            bacteria_data[bid] = {}
        bacteria_data[bid][data_name] = loaded[key]
    return bacteria_data


def crop_around(image, x, y, size=101):
    is_cut = False
    half = size // 2
    x1 = max(0, x - half)
    x2 = min(image.shape[1], x + half + 1)
    y1 = max(0, y - half)
    y2 = min(image.shape[0], y + half + 1)
    if x1 == 0 or x2 == image.shape[1] or y1 == 0 or y2 == image.shape[0]:
        is_cut = True
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image, is_cut


if __name__ == "__main__":
    # mother_dirs = [rf'C:\Users\shsch\OneDrive\Desktop\5\down_{i}' for i in range(0, 6) if i != 4]
    mother_dirs = [rf'C:\Users\shsch\OneDrive\Desktop\5\down_4']
    for current_dir in mother_dirs:
        mes_str = os.path.split(current_dir)[-1]
        # current_dir = r'C:\Users\shsch\OneDrive\Desktop\5\down_2'
        npz_name = 'bacteria_tracks.npz'
        images_dir = os.path.join(current_dir, 'Default')

        single_bac_tracks = load_bac_tracks_from_npz(os.path.join(current_dir, npz_name))
        for bac_id, bac_t in single_bac_tracks.items():
            frames = bac_t['frames']
            xs = bac_t['xs']
            ys = bac_t['ys']
            missing_frames = bac_t['missing']
            images_out_dir = os.path.join(current_dir, 'Cropped_201', f'bac_id_{bac_id}')
            if not os.path.exists(images_out_dir):
                os.makedirs(images_out_dir)
            image_paths = [os.path.join(images_dir, d) for d in os.listdir(images_dir) if d.endswith('tif')]
            for image_path in image_paths:
                img = tiff.imread(image_path)
                image_name = os.path.split(image_path)[-1]
                frame = int(image_name.split('_')[-2][4:])
                if frame not in frames:
                    assert frame in missing_frames
                    continue
                frame_idx = np.where(frames == frame)[0][0]
                bac_x = int(round(xs[frame_idx]))
                bac_y = int(round(ys[frame_idx]))
                cropped, is_cutted = crop_around(img, bac_x, bac_y, 201)
                if is_cutted:
                    print(f'Note! image of frame {frame} is cut for bac id {bac_id} in measurement {mes_str}. Current bac location is: {(bac_x, bac_y)}')
                else:
                    tiff.imwrite(os.path.join(images_out_dir, 'cropped_'+image_name), cropped)
