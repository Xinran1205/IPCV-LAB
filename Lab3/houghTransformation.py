import cv2
import numpy as np


def threshold_image(image, threshold):
    _, threshed = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return threshed


def houghTransformation(thresholdImg, GradiantImg, min_radius, max_radius):
    width = thresholdImg.shape[1]
    height = thresholdImg.shape[0]
    hough_space = np.zeros((height, width, max_radius + 1))
    for y in range(height):
        for x in range(width):
            if thresholdImg[y, x] == 255:
                for r in range(min_radius, max_radius + 1):
                    # for a circle, the middle point is (a, b) and it on the gradiant line of (x, y)
                    a = int(x - r * np.cos(GradiantImg[y, x]))
                    b = int(y - r * np.sin(GradiantImg[y, x]))
                    if a >= 0 and a < width and b >= 0 and b < height:
                        hough_space[b, a, r] += 1
                    a = int(x + r * np.cos(GradiantImg[y, x]))
                    b = int(y + r * np.sin(GradiantImg[y, x]))
                    if a >= 0 and a < width and b >= 0 and b < height:
                        hough_space[b, a, r] += 1
    return hough_space


def find_parameter(hough_space, thresholdH):
    width = hough_space.shape[1]
    height = hough_space.shape[0]
    radius = hough_space.shape[2]
    parameter = []
    for y in range(height):
        for x in range(width):
            for r in range(radius):
                if hough_space[y, x, r] >= thresholdH:
                    parameter.append([y, x, r])
    return parameter


def filter_similar_circles(parameters, center_threshold=20, radius_threshold=20):
    filtered_params = []

    for current_circle in parameters:
        is_similar = False
        for saved_circle in filtered_params:
            center_distance = ((current_circle[0] - saved_circle[0]) ** 2 +
                               (current_circle[1] - saved_circle[1]) ** 2) ** 0.5
            radius_diff = abs(current_circle[2] - saved_circle[2])

            if center_distance < center_threshold and radius_diff < radius_threshold:
                is_similar = True
                break
        # if the circle is not similar to any other circle, add it to the list
        if not is_similar:
            filtered_params.append(current_circle)

    return filtered_params


def draw_circle(image, parameters):
    parameters = filter_similar_circles(parameters)
    for param in parameters:
        cv2.circle(image, (param[1], param[0]), param[2], (0, 0, 255), 2)
    return image


# get the gradiant direction
img = cv2.imread("coins1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
AfterBlur = cv2.GaussianBlur(img, (7, 7), 0)
DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
Direction = np.arctan2(DerivativeY, DerivativeX)

cv2.imwrite("threshold.jpg", threshold_image(cv2.imread("Magnitude.jpg", cv2.IMREAD_GRAYSCALE), 100))
hought_space = houghTransformation(cv2.imread("threshold.jpg", cv2.IMREAD_GRAYSCALE), Direction, 5, 100)
parameter = find_parameter(hought_space, 12)
result = draw_circle(cv2.imread("coins1.png"), parameter)
cv2.imwrite("result.jpg", result)
