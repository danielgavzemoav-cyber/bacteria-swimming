import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import os
from collections import Counter
from scipy.optimize import curve_fit

folders = ['20']  # List of folders to process

distance_threshold = 10
circularity_threshold = 0.7

all_filtered_z = []

for folder_path in folders:
    print(f"\n=== Processing folder: {folder_path} ===")

    # Extract initial height from folder name
    match = re.search(r'(\d+)', os.path.basename(folder_path))
    initial_height = float(match.group(1)) if match else 0
    print(f"Initial height from folder: {initial_height}")

    for filename in os.listdir(folder_path):
        if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
            continue

        filepath = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue

        blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
        detected_rings = []
        detected_contours = []

        for thresh_val in range(1, 201, 10):
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue

            for i, cnt in enumerate(contours):
                if hierarchy[0][i][2] != -1:
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * math.pi * area / (perimeter ** 2)
                    if circularity < circularity_threshold:
                        continue

                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    radius = float(radius)

                    too_close = False
                    for (cx, cy, cr) in detected_rings:
                        dist = math.hypot(cx - x, cy - y)
                        if dist < distance_threshold:
                            if radius > cr:
                                idx = detected_rings.index((cx, cy, cr))
                                detected_rings[idx] = (x, y, radius)
                                detected_contours[idx] = cnt
                            too_close = True
                            break
                    if not too_close:
                        detected_rings.append((x, y, radius))
                        detected_contours.append(cnt)

        # Draw circles and save annotated image
        output_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x, y, radius) in detected_rings:
            z = (-1.21 * radius + 21.172) + initial_height
            lower_limit = max(initial_height - 26, 0)
            if lower_limit <= z <= initial_height:
                all_filtered_z.append(z)

            center = (int(x), int(y))
            cv2.circle(output_circles, center, int(radius), (0, 255, 0), 2)
            cv2.circle(output_circles, center, 3, (0, 0, 255), -1)
            cv2.putText(output_circles, f'{radius:.1f}', (center[0] + int(radius) + 5, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        save_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_with_circles.jpg")
        cv2.imwrite(save_path, output_circles)

# === Plotting Combined Data ===
if not all_filtered_z:
    print("No valid blobs found in the combined folders.")
else:
    min_z = min(all_filtered_z)
    max_z = max(all_filtered_z)
    bin_start = 2 * math.floor(min_z / 2)
    bin_end = 2 * math.ceil(max_z / 2)
    edges = np.arange(bin_start, bin_end + 2, 2)

    def assign_to_lower_bin(z):
        for edge in edges[:-1]:
            if edge <= z < edge + 2:
                return edge
        if z == edges[-1]:
            return edges[-2]
        return None

    mapped_bins = [assign_to_lower_bin(z) for z in all_filtered_z]
    counts = Counter(b for b in mapped_bins if b is not None)

    sorted_bins = sorted(counts.keys())
    counts_per_bin = [counts[b] // 2 for b in sorted_bins]
    from scipy.optimize import curve_fit

    # Convert to numpy arrays
    x = np.array(sorted_bins)
    y = np.array(counts_per_bin)

    # Filter out zero or negative counts (log-scale won't work with those)
    x_filtered = []
    y_filtered = []
    for xi, yi in zip(x, y):
        if yi > 0:
            x_filtered.append(xi)
            y_filtered.append(yi)
    x = np.array(x_filtered)
    y = np.array(y_filtered)


    # Define exponential function
    def exp_func(x, a, b):
        return a * np.exp(b * x)


    # Select fitting ranges (safe for smaller datasets)
    if len(x) >= 7:
        subset1_x = x[-4:]
        subset1_y = y[-4:]
        subset2_x = x[1:7]
        subset2_y = y[1:7]
    else:
        print("Not enough points for both fit ranges.")
        subset1_x = x[-4:]
        subset1_y = y[-4:]
        subset2_x = x[1:]
        subset2_y = y[1:]

    # Combine subsets
    combined_x = np.concatenate([subset1_x, subset2_x])
    combined_y = np.concatenate([subset1_y, subset2_y])

    # Perform exponential fit
    try:
        params, _ = curve_fit(exp_func, combined_x, combined_y, p0=(1, -0.1))
        a_fit, b_fit = params
        print(f"Fit parameters: a = {a_fit:.2e}, b = {b_fit:.2f}")
        x_fit = np.linspace(min(x), max(x), 300)
        y_fit = exp_func(x_fit, a_fit, b_fit)
        fit_label = f'Exp Fit: y = {a_fit:.1e} · e^({b_fit:.2f}·x)'
        plot_fit = True
    except RuntimeError as e:
        print("Fit failed:", e)
        plot_fit = False

    # Plot data and fit
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data')
    if plot_fit:
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=fit_label)

    plt.yscale('log')
    plt.title('E - coli distribution inside the drop (with exponential fit)')
    plt.xlabel('height (micrometer)')
    plt.ylabel('E - coli count')
    plt.xticks(sorted(x))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_bins, counts_per_bin, color='blue')
    plt.yscale('log')
    plt.title('E - coli distibution inside the drop')
    plt.xlabel('height (micrometer)')
    plt.ylabel('E - coli count ')
    plt.xticks(sorted_bins)
    plt.grid(True)
    plt.tight_layout()
    plt.show()