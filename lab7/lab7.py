import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

def match_images(im1,im2,kp1,kp2,des1,des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    if len(good) >= 4:
        pts1 = []
        pts2 = []
        for m in good:
            pts1.append(kp1[m[0].queryIdx].pt)
            pts2.append(kp2[m[0].trainIdx].pt)

        # matrix points
        points1 = np.float32(pts1).reshape(-1, 1, 2)
        points2 = np.float32(pts2).reshape(-1, 1, 2)
    else:
        raise AssertionError("No enough keypoints.")
    
    return points1, points2


def bf_orb(im1,im2, n_show = 20, use_knn=False):
	orb = cv.ORB_create()

	# Find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(im1,None)
	kp2, des2 = orb.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2)

def bf_sift(im1,im2, n_show=20, use_knn=False):
	# Initiate SIFT detector
	sift = cv.SIFT_create()
	
	# Find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(im1,None)
	kp2, des2 = sift.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2)

def bf_surf(im1,im2, n_show=20, use_knn=False):
	# Initiate SURF detector
	surf = cv.xfeatures2d_SURF.create(400)

	# Find the keypoints and descriptors with SIFT
	kp1, des1 = surf.detectAndCompute(im1,None)
	kp2, des2 = surf.detectAndCompute(im2,None)

	return match_images(im1,im2,kp1,kp2,des1,des2)


def apply_homography(img1, img2, transformation_matrix):
    
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Check borders in new perspective
    transformed_corners = cv.perspectiveTransform(corners, transformation_matrix)

    # Determine new rectangle where new image will be
    [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel())

    x_max = max(w2, x_max)
    y_max = max(h2, y_max)

    t_distance = [-x_min, -y_min]

    translation_matrix = np.array([[1, 0, t_distance[0]], [0, 1, t_distance[1]], [0, 0, 1]]) 

    mult_matrix = translation_matrix @ transformation_matrix

    final_size = (x_max - x_min, y_max - y_min)

    combined_img = cv.warpPerspective(img1_resized, mult_matrix, final_size)
    combined_img[t_distance[1]:h2+t_distance[1], t_distance[0]:w2+t_distance[0]] = img2_resized

    return combined_img

# Main code:
img1 = cv.imread(sys.argv[1])
img2 = cv.imread(sys.argv[2])

h1, w1, _ = img1.shape
img1_resized = cv.resize(img1, (400, 400))

h2, w2, _ = img2.shape
img2_resized = cv.resize(img2, (400, 400))

img1_gray = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)

# Find points of interest
points1_sift, points2_sift = bf_sift(img1_gray, img2_gray)
points1_orb, points2_orb = bf_orb(img1_gray, img2_gray)
points1_surf, points2_surf = bf_surf(img1_gray, img2_gray)

# Find homografy with RANSAC
transformation_matrix_sift, inliers_sift = cv.findHomography(points1_sift, points2_sift, cv.RANSAC)
transformation_matrix_orb, inliers_orb = cv.findHomography(points1_orb, points2_orb, cv.RANSAC)
transformation_matrix_surf, inliers_surf = cv.findHomography(points1_surf, points2_surf, cv.RANSAC)

combined_img_sift = apply_homography(img1_resized, img2_resized, transformation_matrix_sift)
combined_img_orb = apply_homography(img1_resized, img2_resized, transformation_matrix_orb)
combined_img_surf = apply_homography(img1_resized, img2_resized, transformation_matrix_surf)

plt.subplot(221).set_ylabel("SIFT"), plt.imshow(cv.cvtColor(combined_img_sift, cv.COLOR_BGR2RGB))
plt.subplot(222).set_ylabel("ORB"), plt.imshow(cv.cvtColor(combined_img_orb, cv.COLOR_BGR2RGB))
plt.subplot(223).set_ylabel("SURF"), plt.imshow(cv.cvtColor(combined_img_surf, cv.COLOR_BGR2RGB))

plt.tight_layout()

plt.show()