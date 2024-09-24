import sys
import cv2 as cv
import numpy as np

image = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

# Create ORB object
orb = cv.ORB_create()

# Detect key points
keypoints = orb.detect(image, None)

# Calculate ORB descriptors
keypoints, descriptors = orb.compute(image, keypoints)

# Draw keypoints
img_orb = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

cv.imshow('ORB', img_orb)

cv.waitKey(0)
cv.destroyAllWindows()
