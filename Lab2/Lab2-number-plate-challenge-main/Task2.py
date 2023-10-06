import cv2
import numpy as np


input_image = cv2.imread("car1.png")

kernel = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]])/9

blur_image = np.zeros(input_image.shape)

#set all pixels to 255
blur_image.fill(255)

height = input_image.shape[0]
width = input_image.shape[1]
channels = input_image.shape[2]

for i in range(1, height-1):
    for j in range(1, width-1):
        # we need to convolution for each channel
        for c in range(channels):
            image_box = input_image[i-1:i+2, j-1:j+2, c]
            blur_image[i, j, c] = np.sum(image_box * kernel)

# this can be calculated directly. Very good!!!!
output_image = input_image+40*(input_image - blur_image)

cv2.imwrite("task2.jpg", output_image)



# This code below are using gaussian kernel to blur the image and then sharpen the image.

# def gaussian_kernel(size, sigma):
#     kernel = np.fromfunction(
#         lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
#                      np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
#         (size, size)
#     )
#     return kernel / kernel.sum()
#
#
# input_image = cv2.imread("car1.png")
# height, width, channels = input_image.shape
# # create a gaussian kernel with size 3*3 and sigma = 0.8
# kernel = gaussian_kernel(3, 0.8)
#
# blur_image = np.zeros_like(input_image)
#
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         for c in range(channels):
#             image_box = input_image[i - 1:i + 2, j - 1:j + 2, c]
#             blur_image[i, j, c] = np.sum(image_box * kernel)
#
# output_image = input_image + 10 * (input_image - blur_image)
#
# cv2.imwrite("task2.jpg", output_image)
