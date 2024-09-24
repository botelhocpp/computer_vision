import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class SeamCarving():
    max_energy = 1000000.0

    def __init__(self, img):
        self.img_arr = img.astype(int)
        self.height, self.width = img.shape[:2]
        self.energy_arr = np.empty((self.height, self.width))
        self.compute_energy_arr()

    def is_border(self, i, j):
        return (i == 0 or i == self.height - 1) or (j == 0 or j == self.width - 1)

    def compute_energy(self, i, j):
        if self.is_border(i, j):
            return self.max_energy

        b = abs(self.img_arr[i - 1, j, 0] - self.img_arr[i + 1, j, 0])
        g = abs(self.img_arr[i - 1, j, 1] - self.img_arr[i + 1, j, 1])
        r = abs(self.img_arr[i - 1, j, 2] - self.img_arr[i + 1, j, 2])

        b += abs(self.img_arr[i, j - 1, 0] - self.img_arr[i, j + 1, 0])
        g += abs(self.img_arr[i, j - 1, 1] - self.img_arr[i, j + 1, 1])
        r += abs(self.img_arr[i, j - 1, 2] - self.img_arr[i, j + 1, 2])

        energy = b + g + r

        return energy

    def swapaxes(self):
        self.energy_arr = np.swapaxes(self.energy_arr, 0, 1)
        self.img_arr = np.swapaxes(self.img_arr, 0, 1)
        self.height, self.width = self.width, self.height

    def compute_energy_arr(self):
        self.energy_arr[[0, -1], :] = self.max_energy
        self.energy_arr[:, [0, -1]] = self.max_energy

        self.energy_arr[1:-1, 1:-1] = np.add.reduce(
            np.abs(self.img_arr[:-2, 1:-1] - self.img_arr[2:, 1:-1]), -1)
        self.energy_arr[1:-1, 1:-1] += np.add.reduce(
            np.abs(self.img_arr[1:-1, :-2] - self.img_arr[1:-1, 2:]), -1)

    def compute_seam(self, horizontal=False):
        if horizontal:
            self.swapaxes()

        energy_sum_arr = np.empty_like(self.energy_arr)

        energy_sum_arr[0] = self.energy_arr[0]
        for i in range(1, self.height):
            energy_sum_arr[i, :-1] = np.minimum(
                energy_sum_arr[i - 1, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i, 1:] = np.minimum(
                energy_sum_arr[i, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i] += self.energy_arr[i]

        seam = np.empty(self.height, dtype=int)
        seam[-1] = np.argmin(energy_sum_arr[-1, :])
        seam_energy = energy_sum_arr[-1, seam[-1]]

        for i in range(self.height - 2, -1, -1):
            l, r = max(0, seam[i + 1] -
                        1), min(seam[i + 1] + 2, self.width)
            seam[i] = l + np.argmin(energy_sum_arr[i, l: r])

        if horizontal:
            self.swapaxes()

        return (seam_energy, seam)

    def carve(self, horizontal=False, seam=None, remove=True):
        if horizontal:
            self.swapaxes()
        
        if seam is None:
            seam = self.compute_seam()[1]
            
        if remove:
            self.width -= 1
        else:
            self.width += 1

        new_arr = np.empty((self.height, self.width, 3))
        new_energy_arr = np.empty((self.height, self.width))
        mp_deleted_count = 0

        for i, j in enumerate(seam):
            if remove:
                if self.energy_arr[i, j] < 0:
                    mp_deleted_count += 1
                new_energy_arr[i] = np.delete(
                    self.energy_arr[i], j)
                new_arr[i] = np.delete(self.img_arr[i], j, 0)
            else:
                new_energy_arr[i] = np.insert(
                    self.energy_arr[i], j, 0, 0)

                new_pixel = self.img_arr[i, j]
                if not self.is_border(i, j):
                    new_pixel = (
                        self.img_arr[i, j - 1] + self.img_arr[i, j + 1]) // 2

                new_arr[i] = np.insert(self.img_arr[i], j, new_pixel, 0)

        self.img_arr = new_arr
        self.energy_arr = new_energy_arr

        for i, j in enumerate(seam):
            for k in range(j - 1, j + 1):
                if 0 <= k < self.width and self.energy_arr[i, k] >= 0:
                    self.energy_arr[i, k] = self.compute_energy(i, k)
        
        if horizontal:
            self.swapaxes()

        return mp_deleted_count

    def resize(self, new_height=None, new_width=None):
        if new_height is None:
            new_height = self.height
        if new_width is None:
            new_width = self.width

        while self.width != new_width:
            self.carve(horizontal=False, remove=self.width > new_width)

        while self.height != new_height:
            self.carve(horizontal=True, remove=self.height > new_height)

    def remove_mask(self, mask):
        mp_count = np.count_nonzero(mask)

        self.energy_arr[mask] *= -(self.max_energy ** 2)
        self.energy_arr[mask] -= (self.max_energy ** 2)

        while mp_count:
            v_seam_energy, v_seam = self.compute_seam(False)
            h_seam_energy, h_seam = self.compute_seam(True)

            horizontal, seam = False, v_seam

            if v_seam_energy > h_seam_energy:
                horizontal, seam = True, h_seam
            
            mp_count -= self.carve(horizontal, seam)


    def image(self):
        return self.img_arr.astype(np.uint8)


def generate_mask(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define various ranges of color
    lower_yellow1 = np.array([20, 100, 100])
    upper_yellow1 = np.array([30, 255, 255])
    mask1 = cv.inRange(hsv_img, lower_yellow1, upper_yellow1)

    lower_yellow2 = np.array([15, 100, 100])
    upper_yellow2 = np.array([20, 255, 255])
    mask2 = cv.inRange(hsv_img, lower_yellow2, upper_yellow2)

    lower_yellow3 = np.array([30, 100, 100])
    upper_yellow3 = np.array([35, 255, 255])
    mask3 = cv.inRange(hsv_img, lower_yellow3, upper_yellow3)

    # Combine all masks
    mask = mask1 + mask2 + mask3

    # Remove small artifacts
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_OPEN, kernel)

    # Filter the mask
    mask_cleaned = cv.GaussianBlur(mask_cleaned, (5, 5), 0)

    return cv.bitwise_not(mask_cleaned)

# Main code:
img = cv.imread(sys.argv[1])

img_resized = cv.resize(img, (500, 500))

gen_mask = generate_mask(img_resized)

cv.imwrite("mask.png", gen_mask)

mask = gen_mask != 255

sc = SeamCarving(img_resized)
sc.remove_mask(mask)

res = sc.image()
cv.imwrite("res.png", res)

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(cv.cvtColor(img_resized, cv.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Mascara')
plt.imshow(mask, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Resultado')
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

cv.waitKey(0)
