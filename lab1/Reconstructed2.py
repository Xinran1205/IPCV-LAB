import cv2
import numpy as np

# 读取图像
image = cv2.imread('mandrill1.jpg')

# 分离图像的三个颜色通道
b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]

#这里用来填充平移后缺失的红色分量
mean_red = np.mean(r)

#保存一个image，这个image没有红色分量
ResultImage = image.copy()
ResultImage[:,:,2] = 0

#遍历每个像素点
for i in range (image.shape[0]-30):
    for j in range (image.shape[1]-30):
        ResultImage[i+30][j+30][2] = image[i][j][2]

for i in range (30):
    for j in range (image.shape[1]):
        ResultImage[i][j][2] = mean_red

for i in range (30,image.shape[0]):
    for j in range (30):
        ResultImage[i][j][2] = mean_red

# 显示原始和平移后的图像
cv2.imshow('Original', image)
cv2.imshow('Red Channel Translated', ResultImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


