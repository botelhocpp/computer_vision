import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

r_kernel = np.array([
    [0, 0, 0],
    [0, 0.3, 0],
    [0, 0, 0]
])
g_kernel = np.array([
    [0, 0, 0],
    [0, 0.59, 0],
    [0, 0, 0]
])
b_kernel = np.array([
    [0, 0, 0],
    [0, 0.11, 0],
    [0, 0, 0]
])

img = cv.imread(sys.argv[1])

r_conv = cv.filter2D(img[:, :, 2], -1, r_kernel)
g_conv = cv.filter2D(img[:, :, 1], -1, g_kernel)
b_conv = cv.filter2D(img[:, :, 0], -1, b_kernel)

res_conv = r_conv + g_conv + b_conv

cv.imwrite('gray.jpg', res_conv)

_, axis = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

axis[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axis[0].set_title('Imagem Original')
axis[0].axis('off')

# Segundo gráfico
axis[1].imshow(res_conv, cmap='gray')
axis[1].set_title('Imagem em Escala de Cinza')
axis[1].axis('off')

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
