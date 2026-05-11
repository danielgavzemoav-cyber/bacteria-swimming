#!/usr/bin/env python3
import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt

def find_bright_blobs(binary_image, min_radius=15):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        if radius >= min_radius:
            blobs.append((cnt, x, y, radius))
    return blobs

def find_blob_in_roi(binary_image, prev_x, prev_y, prev_r, min_radius=15, img_color=None, fname="debug.png", output_folder="output_with_circles"):
    h, w = binary_image.shape
    search_radius = max(150, int(prev_r * 3))

    x1 = max(prev_x - search_radius, 0)
    y1 = max(prev_y - search_radius, 0)
    x2 = min(prev_x + search_radius, w)
    y2 = min(prev_y + search_radius, h)
    roi = binary_image[y1:y2, x1:x2]

    # Always call this
    blobs = find_bright_blobs(roi, min_radius)

    # Optional debug image
    if img_color is not None:
        all_blobs_img = img_color.copy()
        for _, x_rel, y_rel, r in blobs:
            x_abs = x1 + int(x_rel)
            y_abs = y1 + int(y_rel)
            cv2.circle(all_blobs_img, (x_abs, y_abs), int(r), (0, 255, 255), 2)
            cv2.putText(all_blobs_img, f"{r:.1f}", (x_abs + 5, y_abs), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imwrite(os.path.join(output_folder, f"all_blobs_{fname}"), all_blobs_img)

    # Filter out oversized blobs
    max_radius_limit = min(h, w) // 2
    blobs = [b for b in blobs if b[3] < max_radius_limit]

    if not blobs:
        return None, None, None

    def dist(blob):
        _, x_rel, y_rel, _ = blob
        return (x_rel - (prev_x - x1))**2 + (y_rel - (prev_y - y1))**2

    closest_blob = min(blobs, key=dist)
    _, x_rel, y_rel, radius = closest_blob

    x_abs = x1 + int(x_rel)
    y_abs = y1 + int(y_rel)
    return x_abs, y_abs, radius

def extract_number(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else None

def preprocess_image(img_gray, save_path=None):
    blurred = cv2.blur(img_gray, (11, 11))
    _, binary = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    if save_path is not None:
        cv2.imwrite(save_path, binary)
    return binary

def track_blob_and_create_dict(folder_path, min_radius=15, output_folder="new_output"):
    files = [f for f in os.listdir(folder_path) if re.search(r'\d+', f)]
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    files.sort(key=lambda f: extract_number(f))

    if not files:
        print("No image files with numbers found.")
        return {}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    first_path = os.path.join(folder_path, files[0])
    first_gray = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    if first_gray is None:
        print("Failed to load first image.")
        return {}

    processed_first = preprocess_image(first_gray, os.path.join(output_folder, f"blurred_{files[0]}"))
    initial_blobs = find_bright_blobs(processed_first, min_radius)
    if not initial_blobs:
        print("No bright blob found in first image.")
        return {}

    largest_blob = max(initial_blobs, key=lambda b: b[3])
    _, x0, y0, r0 = largest_blob
    x0, y0, r0 = int(x0), int(y0), r0

    print(f"Initial blob at ({x0},{y0}) radius {r0:.2f} in {files[0]}")
    results_dict = {extract_number(files[0]): r0}

    color_img = cv2.imread(first_path)
    cv2.circle(color_img, (x0, y0), int(r0), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, files[0]), color_img)

    prev_x, prev_y, prev_r = x0, y0, r0

    for fname in files[1:]:
        path = os.path.join(folder_path, fname)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Failed to load {fname}")
            results_dict[extract_number(fname)] = None
            continue

        binary_img = preprocess_image(img_gray, os.path.join(output_folder, f"blurred_{fname}"))
        img_color = cv2.imread(path)
        x, y, r = find_blob_in_roi(binary_img, prev_x, prev_y, prev_r, min_radius, img_color=img_color, fname=fname,
                                   output_folder=output_folder)

        if r is None:
            print(f"Blob not found in {fname}")
            results_dict[extract_number(fname)] = None
        else:
            if r < prev_r:
                r = prev_r
                x, y = prev_x, prev_y

            print(f"{fname}: radius {r:.2f} at ({x},{y})")
            results_dict[extract_number(fname)] = r
            prev_x, prev_y, prev_r = x, y, r

            color_img = cv2.imread(path)
            cv2.circle(color_img, (x, y), int(r), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, fname), color_img)

    return results_dict

# === USAGE ===
folder_path = "output_with_circles"  # current folder
results = track_blob_and_create_dict(folder_path)

# Final dictionary
print("\nFinal results dictionary:")
print(results)
# Assume results dict is from your tracking function
# Example: results = {...}  # your dictionary

# Filter data: keys between 18 and 63 inclusive, and non-None values
filtered = [(k, v) for k, v in results.items() if 18 <= k <= 63 and v is not None]

if filtered:
    x = np.array([item[0] for item in filtered])
    y = np.array([item[1] for item in filtered])

    # Linear fit
    m, b = np.polyfit(x, y, 1)
    y_fit = m * x + b

    # Plot
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, color='blue', label='Radius data')
    plt.plot(x, y_fit, color='red', label=f'Linear fit: y={m:.3f}x + {b:.3f}')
    plt.xlabel('height ($\mu$ m)')
    plt.ylabel('Radius ($\mu$ m)')
    #plt.title('Radius vs Fra with Linear Fit (18 ≤ frame ≤ 63)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No valid data points found in the selected frame range for fitting.")