################################################
#
# COMS30068 - colourthr.py
# University of Bristol
#
################################################

import cv2
import numpy as np

# Read image from file
image = cv2.imread("image/mandrillRGB.jpg", 1)

# Threshold by looping through all pixels
th = 200

# if the blue channel of a pixel is greater than 200, then set all three channels (BGR) of the pixel to 255 (white).
# Otherwise, set all three channels to 0 (black).
# Specifically, (Blue > th)*255 returns a Boolean array,
# where pixels greater than the threshold are True and others are False.
# Multiplying True by 255 gives 255, and multiplying False by 255 gives 0.
Blue = image[:, :, 0]
image[:, :, 0] = (Blue > th) * 255
image[:, :, 1] = (Blue > th) * 255
image[:, :, 2] = (Blue > th) * 255

# Or threshold by looping through all pixels
# for y in range(0, image.shape[0]):  # go through all rows (or scanlines)
#	for x in range(0, image.shape[1]):  # go through all columns
#		pixelBlue = image[y, x, 0]
#		pixelGreen = image[y, x, 1]
#		pixelRed = image[y, x, 2]
#		if (pixelBlue>200):
#			image[y, x, 0] = 255
#			image[y, x, 1] = 255
#			image[y, x, 2] = 255
#		else:
#			image[y, x, 0] = 0
#			image[y, x, 1] = 0
#			image[y, x, 2] = 0

# Save thresholded image
cv2.imshow("Thresholded Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
