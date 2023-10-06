import cv2
import numpy as np
import os

# Step 1: read all images
image_dir = "img/landmark"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
images = [cv2.imread(f) for f in image_files]

# Step 2: calculate the median image
# for example, we have 30 pictures. and for each pixel, we calculate the median value of the pixel in 30 pictures.
median_image_data = np.median(images, axis=0).astype(np.uint8)

# Step 3: save the median image
cv2.imwrite("landmark_without_tourists.jpg", median_image_data)
