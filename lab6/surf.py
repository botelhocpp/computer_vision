import sys
import cv2 as cv
import numpy as np

image = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

# Create SURF object
surf = cv.xfeatures2d.SURF_create(hessianThreshold=1000)

# Detect key points e calculate descriptors
keypoints, descriptors = surf.detectAndCompute(image, None)

img_surf = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('SURF', img_surf)

cv.waitKey(0)
cv.destroyAllWindows()
