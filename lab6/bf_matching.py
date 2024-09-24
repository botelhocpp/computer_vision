import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

def match_images(im1,im2,kp1,kp2,des1,des2, n_show = 20, use_knn=False):
	# create BFMatcher object
	# bf = cv.BFMatcher()
	bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)

	if not use_knn:
		# Match descriptors.
		matches = bf.match(des1, des2)

		# Sort them in the order of their distance.
		matches = sorted(matches, key=lambda x: x.distance)

		# Draw first matches.
		img_match = cv.drawMatches(img1, kp1, img2, kp2, matches[:n_show], None,
								   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	else:
		matches = bf.knnMatch(des1, des2, k=2)
		# Apply ratio test
		good = []
		for m, n in matches:
			if m.distance < 0.75 * n.distance:
				good.append([m])

		img_match = cv.drawMatchesKnn(img1, kp1, img2, kp2, good[:n_show], None,
									  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	return img_match


def bf_orb(im1,im2, n_show = 20, use_knn=False):
	orb = cv.ORB_create()

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(im1,None)
	kp2, des2 = orb.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)

def bf_sift(im1,im2, n_show=20, use_knn=False):
	# Initiate SIFT detector
	sift = cv.SIFT_create()
	
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(im1,None)
	kp2, des2 = sift.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)

def bf_surf(im1,im2, n_show=20, use_knn=False):
	# Initiate SURF detector
	surf = cv.xfeatures2d_SURF.create(400)

	# Find the keypoints and descriptors with SIFT
	kp1, des1 = surf.detectAndCompute(im1,None)
	kp2, des2 = surf.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2, n_show =n_show, use_knn=use_knn)

# Main code:
img1 = cv.imread(sys.argv[1])
img2 = cv.imread(sys.argv[2])

img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

im_sift = bf_sift(img1,img2,use_knn=True)
im_orb = bf_orb(img1,img2,use_knn=True)
im_surf = bf_surf(img1,img2,use_knn=True)

plt.subplot(221).set_ylabel("SIFT"), plt.imshow(im_sift,'gray')
plt.subplot(222).set_ylabel("ORB"), plt.imshow(im_orb,'gray')
plt.subplot(223).set_ylabel("SURF"), plt.imshow(im_surf,'gray')

plt.tight_layout()

plt.show()
