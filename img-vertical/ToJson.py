import cv2
import json
import numpy as np
import os

def detect_rectangles(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color thresholds for green and red
    lower_green = np.array([36, 100, 100])
    upper_green = np.array([70, 255, 255])
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create masks
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare JSON data
    data = {"parkings": {}}
    index = 0
    target_set = False

    # Define default dimensions as fractions of the image dimensions
    default_w = 0.04166667
    default_h = 0.12

    # Function to process contours and record data
    def process_contours(contours, is_green):
        nonlocal index, target_set
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                center_x = x + w / 2
                center_y = y + h / 2
                rel_center_x = center_x / width
                rel_center_y = center_y / height

                rectangle_data = [rel_center_x, rel_center_y, default_w, default_h]
                if is_green and not target_set:
                    data["target"] = rectangle_data
                    target_set = True
                data["parkings"][str(index)] = rectangle_data
                index += 1

    # Process contours for green and red rectangles
    process_contours(contours_green, True)
    process_contours(contours_red, False)

    # Ensure 'target' appears before 'parkings'
    ordered_data = {"target": data["target"], "parkings": data["parkings"]}

    # Save JSON
    json_filename = os.path.join(output_path, os.path.basename(image_path).replace('.png', '.json'))
    with open(json_filename, 'w') as file:
        json.dump(ordered_data, file, indent=4)

def process_directory(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):  # Assumes images are in .jpg format
            image_path = os.path.join(input_dir, filename)
            detect_rectangles(image_path, output_dir)


# Usage: Update the paths accordingly
input_directory = os.getcwd()  # Current directory
output_directory = '../dest-new'

process_directory(input_directory, output_directory)
