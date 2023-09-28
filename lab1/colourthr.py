################################################
#
# COMS30068 - colourthr.py
# University of Bristol
#
################################################

import cv2
import numpy as np

# Read image from file
image = cv2.imread("mandrillRGB.jpg", 1)

# Threshold by looping through all pixels
th = 200
#这些代码的意思是，如果某个像素的蓝色通道的值大于200，则将该像素的所有三个通道（BGR）都设置为255（即白色）。否则，将所有三个通道都设置为0（即黑色）。
#具体来说，(Blue > th)*255 会返回一个布尔数组，其中大于阈值的像素为True，其他为False。将True与255相乘得到255，将False与255相乘得到0。
Blue = image[:,:,0]
image[:,:,0] = (Blue > th)*255
image[:,:,1] = (Blue > th)*255
image[:,:,2] = (Blue > th)*255


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
cv2.imwrite("colourthr.jpg", image)