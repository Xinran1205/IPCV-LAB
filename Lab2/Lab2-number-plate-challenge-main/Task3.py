import cv2
import numpy as np

input_image = cv2.imread("car2.png")

output_image = np.zeros(input_image.shape)

output_image.fill(255)

# check if the image is grayscale or color
if len(input_image.shape) == 2:
    print("This is a grayscale image.")
else:
    channels = input_image.shape[2]
    print(f"This image has {channels} channels.")

height = input_image.shape[0]
width = input_image.shape[1]

for i in range(1, height - 1):
    for j in range(1, width - 1):
        # we need to convolution for each channel, median filter size is 3*3
        image_box = input_image[i - 1:i + 2, j - 1:j + 2]
        # normally, np.median() function treats the array as a flattened one-dimensional array and calculates the median.
        output_image[i, j] = np.median(image_box)

cv2.imwrite("task3.jpg", output_image)
