import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Invert Gamora and Nebula colors
def invert_colors(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	img_h = hsv_img[:,:,0]
	img_s = hsv_img[:,:,1]
	img_v = hsv_img[:,:,2]

	width = hsv_img.shape[1]
	height = hsv_img.shape[0]

	for c in range(0, width - 1):
		for l in range(0, height - 1):
			if(img_h[l][c] >= 15 and img_h[l][c] <= 75):
				img_h[l][c] += 65
			elif(img_h[l][c] >= 75 and img_h[l][c] <= 115):
				img_h[l][c] -= 45

	img = cv2.merge([img_h, img_s, img_v])
	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)		
	return img			

# Main code:
filename = sys.argv[1]
or_im = cv2.imread(filename)

im = invert_colors(or_im)

cv2.imwrite('gabula_nemora.jpg', im)

cv2.imshow("Imagem Original", or_im)
cv2.imshow("Imagem Resultante", im)

cv2.waitKey(0)
cv2.destroyAllWindows()
