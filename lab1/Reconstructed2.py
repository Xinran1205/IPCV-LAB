import cv2
import numpy as np

# Read the image
image = cv2.imread('image/mandrill1.jpg')

# Split the image into its channels
b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]

# Calculate the mean of the red channel
mean_red = np.mean(r)

# Create a new image with the same dimensions as the original
ResultImage = image.copy()
# Set the red channel of the new image to 0
ResultImage[:,:,2] = 0

# go through each pixel in the image
for i in range (image.shape[0]-30):
    for j in range (image.shape[1]-30):
        ResultImage[i+30][j+30][2] = image[i][j][2]

for i in range (30):
    for j in range (image.shape[1]):
        ResultImage[i][j][2] = mean_red

for i in range (30,image.shape[0]):
    for j in range (30):
        ResultImage[i][j][2] = mean_red

# Display the image
cv2.imshow('Original', image)
cv2.imshow('Red Channel Translated', ResultImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


