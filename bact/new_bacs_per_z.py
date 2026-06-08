import cv2
import numpy as np
import os
import re
import json
import matplotlib.pyplot as plt
import distance_calibration


def filter_only_circular_cnts(cnts, circ_threshold=0.7, min_area=50):
    circle_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Filter based on circularity
        if circularity > circ_threshold and area > min_area:  # Typical threshold, tune for your images
            circle_cnts.append(cnt)
    print(len(circle_cnts))
    return circle_cnts


def home_made_threshold(gray_image, mean_area=(101, 101), alpha=1.0):
    mean_image = cv2.blur(gray_image, mean_area)
    # Compare original to local mean
    binary = (gray_image > alpha * mean_image).astype(np.uint8) * 255
    return binary

def get_img_radii(image_path):
    radii = []
    img16 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img8 = cv2.convertScaleAbs(img16, alpha=(255.0 / 65535.0))
    img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    # === CLAHE ===
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img_clahe = clahe.apply(img8)
    # === Blur to reduce noise ===
    blurred = cv2.GaussianBlur(img16, (11, 11), 3)
    #_, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    binary = home_made_threshold(blurred, (251, 251), 1.3)
    # === Find contours ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = filter_only_circular_cnts(contours)
    # === Prepare image to draw on ===
    img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    # === Loop over contours ===
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if 200 >= radius >= 10:
            radii.append(radius)
            # Draw circle
            # if radius <= 40:
            #     radius = 110
            cv2.circle(img_color, center, radius, (0, 0, 255), 2)
            cv2.circle(img_color, center, 2, (0, 255, 0), -1)
    # Save the result (with circles drawn) — still 8-bit BGR image now
    new_dir = os.path.join(os.path.dirname(image_path), 'out_track')
    os.makedirs(new_dir, exist_ok=True)
    img_name = os.path.split(image_path)[-1]
    tracked_path = os.path.join(new_dir, img_name)
    sec_tracked_path = os.path.join(new_dir, 'a_'+img_name)
    cv2.imwrite(tracked_path, img_color)
    cv2.imwrite(sec_tracked_path, binary)
    print(radii)
    return radii


def get_distances_from_focus(image_path_list):
    all_radii = []
    for im_p in image_path_list:
        radii = get_img_radii(im_p)
        all_radii += radii
    return all_radii


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



# === MAIN ===
if __name__ == "__main__":
    # mother_path = r"C:\Users\shsch\PycharmProjects\LabCBioPhysics\3"
    # folder_paths = [os.path.join(mother_path, d) for d in os.listdir(mother_path) if d.startswith('processed')]
    # print(folder_paths)
    # calibration_json = "calibration_data.json"  # Your calibration file
    # output_folder = "tracked_output"
    #
    # # Parameters you can tweak
    # min_radius = 8
    # max_radius = 10
    # blur_kernel = (11, 11)
    #
    # for folder_path in folder_paths:
    #     image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]
    #     print(image_paths)
    #     #distances_from_focus = get_distances_from_focus(image_paths)
    #     radii = get_distances_from_focus(image_paths)
    #     print(radii)
    radii = get_img_radii(r'C:\Users\shsch\PycharmProjects\LabCBioPhysics\3\processed_measures_0.01dens_120_mic_from_bottom\10.tif')
    print(radii)

"""
---- another method ----
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of resolution
        minDist=50,  # Minimum distance between detected centers
        param1=40,  # Higher threshold for Canny edge detector
        param2=70,  # Accumulator threshold — smaller detects more circles
        minRadius=20,  # Minimum radius
        maxRadius=100  # Maximum radius
    )
    if circles is None:
        return []
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        radii.append(radius)
        # Draw the outer circle in red and middle in green
        cv2.circle(img_color, center, radius, (0, 0, 255), 2)
        cv2.circle(img_color, center, 2, (0, 255, 0), 3)
"""