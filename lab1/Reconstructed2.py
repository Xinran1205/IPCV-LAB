import cv2
import numpy as np

# 读取图像
image = cv2.imread('mandrill1.jpg')

# 分离图像的三个颜色通道
b, g, r = cv2.split(image)

#这里用来填充平移后缺失的红色分量
mean_red = np.mean(r)

# 定义平移矩阵
# [1 0 tx]
# [0 1 ty]
tx, ty = 30, 30  # 这里你可以修改平移的距离
M = np.float32([[1, 0, tx], [0, 1, ty]])

# 使用 warpAffine 对红色分量进行平移
r_translated = cv2.warpAffine(r, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=mean_red)

# 合并平移后的红色分量和其他两个未变的颜色通道
result = cv2.merge([b, g, r_translated])

# 显示原始和平移后的图像
cv2.imshow('Original', image)
cv2.imshow('Red Channel Translated', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


