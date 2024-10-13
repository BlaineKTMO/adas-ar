import cv2
import numpy as np

# Load the image
image = cv2.imread('input_image.jpg')

# Define the source points (corners of the image)
h, w = image.shape[:2]
src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# Define the destination points (where you want to map the source points)
dst_points = np.float32([[100, 300], [500, 300], [100, 700], [500, 700]])

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to the image
warped_image = cv2.warpPerspective(image, matrix, (w, h))

# Display the original and warped images
cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()