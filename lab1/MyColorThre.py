import cv2 as cv
import numpy as np

image = cv.imread("image/mandrillRGB.jpg")

# A color range in BGR format is defined.
# The lower_bound and upper_bound represent the minimum and maximum values for the blue,
# green, and red channels, respectively.
# The given bounds will match any color where the blue channel is between 200 and 255,
# while the green and red channels can be any value between 0 and 255.
lower_bound = np.array([200, 0, 0])  # BGR format: Blue > 200, Green and Red can be any value between 0 and 255
upper_bound = np.array([255, 255, 255])

#
# The cv.inRange function checks for every pixel in the image if it lies within the defined color range.
# It returns a binary mask where pixels within the range are set to 255 white and
# pixels outside the range are set to 0 (black).
mask = cv.inRange(image, lower_bound, upper_bound)


binary_image = cv.merge([mask, mask, mask])

# if the pixels mask is 255, then set all three channels (BGR) of the pixel to 255 (white).
# this is the same as the above merge function
# result = np.zeros_like(image)
# result[mask == 255] = [255, 255, 255]  # 设置为白色

cv.imshow("Binary Image", binary_image)
cv.waitKey(0)
cv.destroyAllWindows()