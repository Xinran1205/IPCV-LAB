import cv2
import numpy as np


frames = []

for i in range(1, 31):
    # {:05} means the digits must be length of 5, if not, fill with 0
    filename = 'img/waterfall/{:05}.png'.format(i)
    frame = cv2.imread(filename)
    frames.append(frame)

# this is just the easiest way to find the Long-exposure Photography, we just use the mean value of all frames.
avg_frame = np.mean(frames, axis=0).astype(np.uint8)

cv2.imwrite('long_exposure_simulation.jpg', avg_frame)