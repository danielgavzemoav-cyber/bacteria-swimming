import numpy as np
import cv2
import os


def get_dir_pics_paths(mdpath):
    dirs = [os.path.join(mdpath, d) for d in os.listdir(mdpath) if os.path.isdir(os.path.join(mdpath, d))]
    if len(dirs) != 1 and all(['Default' != os.path.split(d)[-1] for d in dirs]):
        dirs = [os.path.join(d, 'Default') for d in dirs]
        if not all([os.path.exists(d) for d in dirs]):
            print(f'no Default in some of the subdirs in {dirs}')
            return None

    raw_images_paths = []
    for d in dirs:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        raw_images_paths += files

    return raw_images_paths


def adjust_image(img):
    p_low, p_high = np.percentile(img, (1, 99))

    # Clip and stretch
    img_auto = np.clip(img, p_low, p_high)
    img_auto = (img_auto - p_low) / (p_high - p_low) * 65535.0

    # Convert back to uint16
    img_auto_uint16 = img_auto.astype(np.uint16)
    return img_auto_uint16


def preprocess_dir_pics(mother_dir, output_dir_path):
    images_paths = get_dir_pics_paths(mother_dir)
    print(images_paths)
    print('\n')
    for i, image in enumerate(images_paths):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        processed_img = adjust_image(img)
        out_img_path = os.path.join(output_dir_path, f'{i}.tif')
        print(out_img_path)
        cv2.imwrite(out_img_path, processed_img)


exp_dir = r'C:\Users\shsch\PycharmProjects\LabCBioPhysics\3'
mother_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d)) and 'processed' not in d]
for md in mother_dirs:
    in_dir_path = os.path.join(exp_dir, md)
    out_dir_name = 'processed_' + os.path.split(md)[-1]
    out_dir_path = os.path.join(exp_dir, out_dir_name)
    os.makedirs(out_dir_path, exist_ok=True)
    preprocess_dir_pics(in_dir_path, out_dir_path)
