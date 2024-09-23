import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sepia_kernel = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

img = cv.imread(sys.argv[1])

sepia_img = cv.transform(img, sepia_kernel)

cv.imwrite('sepia.jpg', sepia_img)

_, axis = plt.subplots(1, 2, figsize=(12, 12), sharex=True, sharey=True)

axis[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axis[0].set_title('Imagem Original')
axis[0].axis('off')

axis[1].imshow(cv.cvtColor(sepia_img, cv.COLOR_BGR2RGB))
axis[1].set_title('Imagem Sépia')
axis[1].axis('off')

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
