import cv2
import numpy as np
import lane_modules as lm

# Import base image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Canny image
canny_image = lm.canny(lane_image)

# Cropped and masked canny image
cropped_image = lm.region_of_interest(canny_image)

# Create Hough Transform to recognize lines from the cropped image
lines = lm.hough_transform(cropped_image)

# Create averaged lines
averaged_lines = lm.average_slope_intercept(lane_image, lines)

# Line image on black background
line_image = lm.display_lines(lane_image, averaged_lines)

# Blend the base image with line image
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow('result', combo_image)
cv2.waitKey(10000)