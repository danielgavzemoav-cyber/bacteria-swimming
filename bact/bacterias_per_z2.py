import cv2
import numpy as np
import os
import re
import json
import matplotlib.pyplot as plt

def extract_number(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else None

def subtract_average_and_threshold(img_gray):
    avg_gray = np.mean(img_gray)
    result = img_gray.astype(np.float32) - avg_gray
    result[result < 0] = 0  # Set values below average to zero
    return result.astype(np.uint8)

def preprocess_image(img_gray, blur_kernel=(11,11), threshold_value=1400):
    blurred = cv2.blur(img_gray, blur_kernel)
    _, binary = cv2.threshold(blurred, threshold_value, 65535, cv2.THRESH_BINARY)
    return binary

def find_almost_circular_blobs(binary_image, min_radius=15, max_radius=250, circularity_threshold=0.7):
    """
    Find contours that are almost circular, including nested contours, ignoring large ones.

    Parameters:
    - binary_image: input binary image
    - min_radius: minimum radius of enclosing circle
    - max_radius: maximum radius of enclosing circle
    - circularity_threshold: float between 0 and 1 (1 is perfect circle)

    Returns:
    - blobs: list of (x, y, r) of accepted blobs
    """

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []

    for i, cnt in enumerate(contours):
        # Skip too small contours early
        if cv2.contourArea(cnt) < 10:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < min_radius or r > max_radius:
            continue

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # circularity close to 1 means close to circle
        if circularity >= circularity_threshold:
            blobs.append((int(x), int(y), r))

    return blobs, contours


def load_calibration(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    # Convert keys to int, remove None values
    return {int(k): v for k, v in data.items() if v is not None}

def match_radius_to_distance(radius, calibration_dict):
    # calibration_dict: distance -> radius
    # Return distance with closest radius to detected radius
    closest_dist = min(calibration_dict.keys(), key=lambda d: abs(calibration_dict[d] - radius))
    return closest_dist

def process_single_file(folder_path, fname, calibration_dict=None, output_folder="tracked_output",
                          min_radius=15, threshold=30000, blur_kernel=(11,11)):
    img_path = os.path.join(folder_path, fname)
    img_gray = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_gray is None:
        print(f"Failed to load {fname}, skipping.")
        return None
    all_file_blobs = {}
    #for threshold in range(100, 141, 10):
    binary = preprocess_image(img_gray, blur_kernel, threshold)
    binary_8bit = (binary > 0).astype('uint8') * 255
    blobs, contours = find_almost_circular_blobs(binary_8bit, min_radius=min_radius, max_radius=250, circularity_threshold=0.5)

    # color_img = cv2.imread(img_path)
    # if color_img is None:
    #     print(f"Failed to load {fname} in color, skipping.")
    #     continue

    distance_counts = {}
    binary_color = cv2.cvtColor(binary_8bit, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in blobs:
        dist = r
        # dist = match_radius_to_distance(r, calibration_dict)
        distance_counts[dist] = distance_counts.get(dist, 0) + 1
        all_file_blobs[(x,y)] = dist
        # Draw circle

        cv2.circle(binary_color, (x, y), int(r), (0, 255, 0), 2)
        cv2.putText(binary_color, f"{dist}um", (x - 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.drawContours(binary_color, contours, -1, (0, 0, 255), 2)
    out_fname = fname.split('.')[0] + 'tresh' + str(threshold) + '.' + fname.split('.')[1]
    print(out_fname)
    output_path = os.path.join(output_folder, out_fname)
    cv2.imwrite(output_path, binary_color)
    print(f"Processed {fname} in threshold {threshold}: {len(blobs)} blobs found.")

    ### [[(x1,y1,r1), (x2, y2, r2)], [(x1*,y1*,r1*), (x2*, y2*, r2*)], []]
    return distance_counts


def process_folder_images(folder_path, calibration_json, output_folder="tracked_output",
                          min_radius=15, threshold=140, blur_kernel=(11,11)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #calibration_dict = load_calibration(calibration_json)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    print(files)
    files = [files[0]]
             # and extract_number(f) is not None]
    #files.sort(key=extract_number)[0]

    all_counts = {}  # distance -> total blob count

    for fname in files:
        file_counts = process_single_file(folder_path, fname)

        # Update global counts
        for dist, cnt in file_counts.items():
            all_counts[dist] = all_counts.get(dist, 0) + cnt


    # Plot histogram for all images combined
    distances = sorted(all_counts.keys())
    counts = [all_counts[d] for d in distances]

    plt.figure(figsize=(8,5))
    plt.bar(distances, counts, width=3, color='skyblue', edgecolor='black')
    plt.xlabel("Estimated Distance from Focus (µm)")
    plt.ylabel("Number of Blobs")
    plt.title("Blob Count per Estimated Distance (All Images)")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    return all_counts

# === MAIN ===
if __name__ == "__main__":
    folder_path = r"C:\Users\shsch\PycharmProjects\LabCBioPhysics\2\lower_bound_10_8s\lower+104_bound_11_2s_processed"  # Folder with images
    calibration_json = "calibration_data.json"  # Your calibration file
    output_folder = "tracked_output"

    # Parameters you can tweak
    min_radius = 8
    #threshold = 140
    threshold = 140
    blur_kernel = (11, 11)

    counts = process_folder_images(folder_path, calibration_json,
                                   output_folder=output_folder,
                                   min_radius=min_radius,
                                   threshold=threshold,
                                   blur_kernel=blur_kernel)

    print("\nFinal blob counts by estimated distance:")
    for dist, cnt in sorted(counts.items()):
        print(f"{dist} µm: {cnt} blobs")