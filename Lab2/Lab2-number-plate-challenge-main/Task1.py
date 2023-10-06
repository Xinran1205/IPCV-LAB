import cv2
import numpy as np

input_image = cv2.imread("mandrill.jpg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

kernel = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])/9


output_image = np.zeros(input_image.shape)
#set all pixels to 255
output_image.fill(255)

height = input_image.shape[0]
width = input_image.shape[1]

for i in range(1,height-1):
    for j in range(1,width-1):
        image_box = input_image[i-1:i+2,j-1:j+2]
        output_image[i,j] = np.sum(image_box*kernel)


# output_image = output_image.astype(np.uint8)
# cv2.imshow("Original", input_image)
cv2.imwrite("task1.jpg", output_image)


