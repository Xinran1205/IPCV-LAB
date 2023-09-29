import cv2 as cv

image = cv.imread("image/mandrill.jpg", 1)
# convert to gray scale image
Graycode = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# loop through all pixels and set the pixel value to 255 if it is greater than 128, otherwise set it to 0
# 128 is the threshold value
for i in range(0, Graycode.shape[0]):
    for j in range(0, Graycode.shape[1]):
        if Graycode[i][j] > 128:
            Graycode[i][j] = 255
        else:
            Graycode[i][j] = 0

cv.imshow("Graycode", Graycode)
cv.waitKey(0)
cv.destroyAllWindows()

# This is the same as the above code but using the cv.threshold function
# opencvThreshold = cv.threshold(Graycode, 128, 255, cv.THRESH_BINARY)
# cv.imshow("opencvThreshold", opencvThreshold[1])
# cv.waitKey(0)
# cv.destroyAllWindows()
