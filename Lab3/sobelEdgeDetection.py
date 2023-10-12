import cv2
import numpy as np

# def sobelEdgeDetect (img):
#     # 1. Convert the image to grayscale
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 2. Blur the image using a Gaussian filter
#     width = img.shape[0]
#     height = img.shape[1]
#     AfterBlur = cv2.GaussianBlur(img, (3, 3), 0)
#
#     # 3. Image containing the derivative in the x direction
#     DerivativeX = np.zeros_like(img, dtype=np.float32)
#     kernelX = np.array([[-1, 0, 1],
#                         [-2, 0, 2],
#                         [-1, 0, 1]])
#     for i in range(1, width-1):
#         for j in range(1, height-1):
#             image_box = AfterBlur[i-1:i+2, j-1:j+2]
#             DerivativeX[i, j] = np.sum(image_box * kernelX)
#
#     #use cv2.normalize to map the derivative value to 0 - 255
#     DerivativeX_norm = cv2.normalize(DerivativeX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imwrite("DerivativeX.jpg", DerivativeX_norm)


#     # 4. Image containing the derivative in the y direction
#     DerivativeY = np.zeros_like(img, dtype=np.float32)
#     kernelY = np.array([[-1, -2, -1],
#                         [0, 0, 0],
#                         [1, 2, 1]])
#     for i in range(1, width-1):
#         for j in range(1, height-1):
#             image_box = AfterBlur[i-1:i+2, j-1:j+2]
#             DerivativeY[i, j] = np.sum(image_box * kernelY)
#     DerivativeY_norm = cv2.normalize(DerivativeY, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imwrite("DerivativeY.jpg", DerivativeY_norm)
#
#     # 5. Image containing the magnitude of the gradient
#     Magnitude = np.zeros_like(img)
#     for i in range(1, width-1):
#         for j in range(1, height-1):
#             Magnitude[i, j] = np.sqrt(DerivativeX[i, j]**2 + DerivativeY[i, j]**2)
#     cv2.imwrite("Magnitude.jpg", Magnitude)
#
#     # 6. Image containing the direction of the gradient
#     Direction = np.zeros_like(img, dtype=np.float32)
#     for i in range(1, width - 1):
#         for j in range(1, height - 1):
#             Direction[i, j] = np.arctan2(DerivativeY[i, j], DerivativeX[i, j])
#     # use cv2.normalize to map the direction value to 0 - 255
#     normalizedDirection = (((Direction + np.pi) / (2 * np.pi)) * 255).astype(np.uint8)
#
#     cv2.imwrite("Direction.jpg", normalizedDirection)
#
#
# sobelEdgeDetect(cv2.imread("coins1.png"))

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    non_max = np.zeros((M, N), dtype=np.float32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q, r = 255, 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    non_max[i, j] = gradient_magnitude[i, j]
                else:
                    non_max[i, j] = 0
            except IndexError:
                pass
    return non_max


def double_thresholding(img, low_ratio=0.03, high_ratio=0.2):
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    strong, weak = 255, 50

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result


def sobelEdgeDetect(img):
    # 1. Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Blur the image using a Gaussian filter
    AfterBlur = cv2.GaussianBlur(img, (7, 7), 0)

    # 3. calculate x direction derivative
    DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
    cv2.imwrite("DerivativeX.jpg", np.clip(DerivativeX + 128, 0, 255))

    # 4. calculate y direction derivative
    DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
    cv2.imwrite("DerivativeY.jpg", np.clip(DerivativeY + 128, 0, 255))

    # 6.
    Direction = np.arctan2(DerivativeY, DerivativeX)

    # 5. calculate gradient magnitude
    Magnitude = np.sqrt(DerivativeX**2 + DerivativeY**2)
    normalized_magnitude = (Magnitude / np.max(Magnitude) * 255).astype(np.uint8)
    normalized_magnitude = non_maximum_suppression(normalized_magnitude, Direction)
    normalized_magnitude = double_thresholding(normalized_magnitude)
    cv2.imwrite("Magnitude.jpg", normalized_magnitude)




sobelEdgeDetect(cv2.imread("coins1.png"))