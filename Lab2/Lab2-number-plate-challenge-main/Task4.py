import cv2
import numpy as np
import deconvolution


input_image = cv2.imread("car3.png")

# convert to grayscale, because this wienerDeconvolution function only accept grayscale image
input_image1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

output_image = deconvolution.WienerDeconvoluition(input_image1, 30,  1, 0.01, True)

cv2.imwrite("task4.jpg", output_image)