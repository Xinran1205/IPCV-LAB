import cv2 as cv


image = cv.imread("mandrill.jpg", 1)
Graycode = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# for i in range(0, Graycode.shape[0]):
#     for j in range(0, Graycode.shape[1]):
#         if Graycode[i, j] > 128:
#             Graycode[i, j] = 255
#         else:
#             Graycode[i, j] = 0

# cv.imshow("Graycode", Graycode)
# cv.waitKey(0)
# cv.destroyAllWindows()

#这个是openCV的库包，做的事情和上面是一样的
opencvThreshold = cv.threshold(Graycode, 128, 255, cv.THRESH_BINARY)
cv.imshow("opencvThreshold", opencvThreshold[1])
cv.waitKey(0)
cv.destroyAllWindows()