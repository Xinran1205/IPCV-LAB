import cv2 as cv
import matplotlib.pyplot as plt

# set the backend of matplotlib
import matplotlib
matplotlib.use('TkAgg')

# read the image
mandrill0 = cv.imread('image/mandrill0.jpg')

# color channels
colors = ('b', 'g', 'r')

# for each color channel, calculate the histogram and plot it
for i, color in enumerate(colors):
    hist = cv.calcHist([mandrill0], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)

plt.title('Histogram for color scale picture')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

# Split the image into its channels
b1 = mandrill0[:,:,0]
g1 = mandrill0[:,:,1]
r1 = mandrill0[:,:,2]

swapped_image = mandrill0.copy()
swapped_image[:,:,0] = r1
swapped_image[:,:,1] = b1
swapped_image[:,:,2] = g1

cv.imshow('Swapped image', swapped_image)
cv.waitKey(0)
cv.destroyAllWindows()