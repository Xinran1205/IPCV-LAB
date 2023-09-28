import cv2 as cv
import numpy as np

image = cv.imread("mandrillRGB.jpg")

#这个用的是openCV的inRange方法做的

# 2. 定义颜色范围
lower_bound = np.array([200, 0, 0])  # BGR format: Blue > 200, Green and Red can be any value between 0 and 255
upper_bound = np.array([255, 255, 255])

# 3. 使用inRange函数
mask = cv.inRange(image, lower_bound, upper_bound)

# 4. 创建二值图像
binary_image = cv.merge([mask, mask, mask])
#4 和这个merge是一样的，使用掩码设置相应的像素为白色
# result = np.zeros_like(image)
# result[mask == 255] = [255, 255, 255]  # 设置为白色

# 5. 保存图像
cv.imwrite("colourthr_inrange.jpg", binary_image)