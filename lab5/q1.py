import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color

def calculate_energy_hor(image):
    gray_image = color.rgb2gray(image)
    energy = np.abs(filters.sobel_v(gray_image))
    return energy

def calculate_energy_ver(image):
    gray_image = color.rgb2gray(image)
    energy = np.abs(filters.sobel_h(gray_image))
    return energy

def find_seam_hor(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for j in range(1, c):
        for i in range(r):
            if i == 0:
                idx = np.argmin(M[i:i+2, j-1])
                backtrack[i, j] = idx + i
                min_energy = M[idx + i, j-1]
            else:
                idx = np.argmin(M[i-1:i+2, j-1])
                backtrack[i, j] = idx + i - 1
                min_energy = M[idx + i - 1, j-1]
            M[i, j] += min_energy

    return M, backtrack

def find_seam_ver(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            M[i, j] += min_energy

    return M, backtrack

def remove_seam_hor(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r - 1, c, 3), dtype=image.dtype)
    i = np.argmin(backtrack[:, -1])
    for j in reversed(range(c)):
        output[:, j, 0] = np.delete(image[:, j, 0], [i])
        output[:, j, 1] = np.delete(image[:, j, 1], [i])
        output[:, j, 2] = np.delete(image[:, j, 2], [i])
        i = backtrack[i, j]
    return output

def remove_seam_ver(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(backtrack[-1])
    for i in reversed(range(r)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        j = backtrack[i, j]
    return output

def seam_carving(image, num_seams):
    for _ in range(num_seams):
        energy = calculate_energy_hor(image)
        M, backtrack = find_seam_hor(energy)
        image = remove_seam_hor(image, backtrack)

    for _ in range(num_seams):
        energy = calculate_energy_ver(image)
        M, backtrack = find_seam_ver(energy)
        image = remove_seam_ver(image, backtrack)
    return image

# Main code:
img = io.imread(sys.argv[1])

new_image = seam_carving(img, 20)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(new_image)
ax[1].set_title('Seam Carved Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()
