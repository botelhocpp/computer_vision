import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

kernel = np.array([
    [0.3, 0.59, 0.11],
    [0.3, 0.59, 0.11],
    [0.3, 0.59, 0.11]
])

img = cv.imread(sys.argv[1])

res_conv = cv.transform(img, kernel)

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
