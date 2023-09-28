import cv2 as cv
import numpy as np

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 为matplotlib设置后端
import matplotlib
matplotlib.use('TkAgg')

# 读取图像
mandrill0 = cv.imread('mandrill0.jpg')
#交换颜色通道
b, g, r = cv.split(mandrill0)
swapped_image = cv.merge((b, r, g))
cv.imshow('Swapped image', swapped_image)
cv.waitKey(0)
cv.destroyAllWindows()