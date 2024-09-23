import sys
import cv2 as cv
import numpy as np

def gamma_correction(img, gamma, c = 1.0):
    return 255.0*(c*(img/255.0)**(1.0/gamma))

def turn_yellowish(img, gamma):
    img_copy = img.copy()
    img_g, img_b, img_r = cv.split(img_copy)

    img_r[:,:] = gamma_correction(img_r[:,:], gamma, 1.2) 
    img_g[:,:] = gamma_correction(img_g[:,:], gamma, 1.2) 
    img_b[:,:] = gamma_correction(img_b[:,:], gamma, 0.4) 

    return cv.merge((img_b, img_g, img_r))

# Main code:
img = cv.imread(sys.argv[1])

img_yellow = turn_yellowish(img, 0.5)

cv.imwrite('yellow_jet.jpg', img_yellow)

cv.imshow('Original Jet', img)
cv.imshow('Yellow Jet', img_yellow)

cv.waitKey(0)
cv.destroyAllWindows()
