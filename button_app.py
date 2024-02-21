# import libraries
import os
import sys
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from openvino.runtime import Core
from picamera2 import Picamera2
from time import sleep
import matplotlib

# initiate the camera and define the location of the images
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
image_sample = "data/cars.jpg"

# Define directories for our models
base_dir = "models"
detection_model = "vehicle-detection-0200.xml"
recognition_model = "vehicle-attributes-recognition-barrier-0039.xml"
detection_path = os.path.join(base_dir, detection_model)
recognition_path= os.path.join(base_dir, recognition_model)

## Initialize runtime and include this for the terminal
ie = Core()
matplotlib.use('GTK3Agg')

# Load the model
def model_init(model_path: str) -> Tuple:
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model

# Get attributes
input_key_de, output_keys_de, compiled_model_de = model_init(detection_path)
input_key_re, output_keys_re, compiled_model_re = model_init(recognition_path)

#Get input size
height_de, width_de = list(input_key_de.shape)[2:]
height_re, width_re = list(input_key_re.shape)[2:]

# Take picture
def capture():
    picam2.start()
    print("image processing")
    sleep(3)
    picam2.capture_file(image_sample)
    picam2.stop()
    print("success")

# Read image
def image_show():
    image_de = cv2.imread(image_sample)
    return image_de

# Resize image
def resize_image():
    resized_image_de = cv2.resize(image_show(), (width_de, height_de))
    return resized_image_de

# Expand the dimensions of the image
def input_image():
    input_image_de = np.expand_dims(resize_image().transpose(2, 0, 1), 0)
    return input_image_de

#Show image
def plt_show(raw_image):
    print("rendering image")
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(raw_image)
    plt.savefig('car-gui-detect.jpg')
    print("success")

# Delete unused dimensions and filter out unused results
def box_car():
    boxes = compiled_model_de([input_image()])[output_keys_de]
    boxes = np.squeeze(boxes, (0, 1))
    boxes = boxes[~np.all(boxes==0, axis=1)]
    return boxes

# Filter out low confidence results by cropping image
def crop_images(bgr_image, resized_image, boxes, threshold=0.6) -> np.ndarray:
    # Fetch image shapes to calculate ratio
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Find the boxes ratio
    boxes = boxes[:, 2:]
    # Store the vehicle's position
    car_position = []
    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[0]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image,
            # we position upper box bar little bit lower to make it visible on image
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2
                else int(corner_position * ratio_x * resized_x)
                for idx, corner_position in enumerate(box[1:])
            ]

            car_position.append([x_min, y_min, x_max, y_max])

    return car_position

# Crop the image based on the location of the car
def car():
    car_position = crop_images(image_show(), resize_image(), box_car())
    return car_position

# Find car position
def position():
    pos = car()[0]
    return pos

# Crop the image
def test():
    test_car = image_show()[position()[1]:position()[3], position()[0]:position()[2]]
    return test_car

# Resize the test image given
def resize_image_re():
    image_re = cv2.resize(test(), (width_re, height_re))
    return image_re

# Expand the dimensions of the resized image
def input_image_re():
    input_re = np.expand_dims(resize_image_re().transpose(2, 0, 1), 0)
    return input_re

# Recognize vehicle
def vehicle_recognition(compiled_model_re, input_size, raw_image):
    colors = ['White', 'Gray', 'Yellow', 'Red', 'Green', 'Blue', 'Black']
    types = ['Car', 'Bus', 'Truck', 'Van']
    resized_image_re = cv2.resize(raw_image, input_size)
    input_image_re = np.expand_dims(resize_image_re().transpose(2, 0, 1), 0)
    # Run inference and predict result
    predict_colors = compiled_model_re([input_image_re])[compiled_model_re.output(1)]
    predict_colors = np.squeeze(predict_colors, (2, 3))
    predict_types = compiled_model_re([input_image_re])[compiled_model_re.output(0)]
    attr_color, attr_type = (colors[np.argmax(predict_colors)], types[np.argmax(predict_types)])
    return attr_color, attr_type

# Combine and show
def convert_result(compiled_model_re, bgr_image, resized_image, boxes, threshold=0.6):
    # Define colors
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    # Convert base image from bgr to rgb
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    ## find the position
    car_position = crop_images(image_show(), resized_image, boxes)
    for x_min, y_min, x_max, y_max in car_position:
        # Run inference
        attr_color, attr_type = vehicle_recognition(compiled_model_re, (72, 72), image_show()[y_min:y_max, x_min:x_max])
        # Close window
        plt.close()
        # Print attributes
        rgb_image = cv2.putText(
            rgb_image,
            f"{attr_color} {attr_type}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            colors["green"],
            10,
            cv2.LINE_AA
        )
    return rgb_image


        