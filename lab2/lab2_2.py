import sys
import numpy as np
import cv2 as cv

def create_head(circle, width, height):
    circle_scale = np.float32([[1,0,0],[0,1,0]])

    head = cv.bitwise_not(circle)
    head = cv.warpAffine(head, circle_scale, (width*3,height*3))

    translation_matrix = np.float32([[1,0,100],[0,1,0]])
    translated_img = cv.warpAffine(head,translation_matrix,(width*3, height*3))

    return cv.bitwise_not(translated_img)

def create_body(line, width, height):
    line_scale = np.float32([[1,0,0],[0,1,0]])

    body = cv.bitwise_not(line)
    body = cv.warpAffine(body, line_scale, (width*3,height*3))

    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), 90, 1)
    body = cv.warpAffine(body, rotation_matrix, (width*3,height*3))

    translation_matrix = np.float32([[1,0,100],[0,1,62]])
    body = cv.warpAffine(body,translation_matrix,(width*3,height*3))

    return cv.bitwise_not(body)

def create_right_arm(line, width, height):
    line_scale = np.float32([[.75,0,0],[0,.75,0]])

    right_arm = cv.bitwise_not(line)
    right_arm = cv.warpAffine(right_arm, line_scale, (width*3,height*3))

    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), -45, 1)
    right_arm = cv.warpAffine(right_arm, rotation_matrix, (width*3,height*3))

    translation_matrix = np.float32([[1,0,120],[0,1,79]])
    right_arm = cv.warpAffine(right_arm,translation_matrix,(width*3,height*3))

    return cv.bitwise_not(right_arm)

def create_left_arm(line, width, height):
    line_scale = np.float32([[.75,0,0],[0,.75,0]])

    left_arm = cv.bitwise_not(line)
    left_arm = cv.warpAffine(left_arm, line_scale, (width*3,height*3))

    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), 45, 1)
    left_arm = cv.warpAffine(left_arm, rotation_matrix, (width*3,height*3))

    translation_matrix = np.float32([[1,0,95],[0,1,62]])
    left_arm = cv.warpAffine(left_arm,translation_matrix,(width*3,height*3))

    return cv.bitwise_not(left_arm)

def create_right_leg(line, width, height):
    line_scale = np.float32([[0.75,0,0],[0,0.75,0]])

    right_led = cv.bitwise_not(line)
    right_led = cv.warpAffine(right_led, line_scale, (width*3,height*3))

    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), -45, 1)
    right_led = cv.warpAffine(right_led, rotation_matrix, (width*3,height*3))

    translation_matrix = np.float32([[1,0,119],[0,1,140]])
    right_led = cv.warpAffine(right_led,translation_matrix,(width*3,height*3))

    return cv.bitwise_not(right_led)

def create_left_leg(line, width, height):
    line_scale = np.float32([[0.75,0,0],[0,0.75,0]])

    left_leg = cv.bitwise_not(line)
    left_leg = cv.warpAffine(left_leg, line_scale, (width*3,height*3))

    rotation_matrix = cv.getRotationMatrix2D((width/2, height/2), 45, 1)
    left_leg = cv.warpAffine(left_leg, rotation_matrix, (width*3,height*3))

    translation_matrix = np.float32([[1,0,95],[0,1,125]])
    left_leg = cv.warpAffine(left_leg,translation_matrix, (width*3,height*3))

    return cv.bitwise_not(left_leg)

# Main code:
circle = cv.imread('img/circle.jpg')
circle_width = circle.shape[1]
circle_height = circle.shape[0]

line = cv.imread('img/line.jpg')
line_width = line.shape[1]
line_height = line.shape[0]

head = create_head(circle, circle_width, circle_height)
body = create_body(line, line_width, line_height)
right_arm = create_right_arm(line, line_width, line_height)
left_arm = create_left_arm(line, line_width, line_height)
right_leg = create_right_leg(line, line_width, line_height)
left_leg = create_left_leg(line, line_width, line_height)

img_sticky = cv.bitwise_and(head, body)
img_sticky = cv.bitwise_and(img_sticky, right_arm)
img_sticky = cv.bitwise_and(img_sticky, left_arm)
img_sticky = cv.bitwise_and(img_sticky, right_leg)
img_sticky = cv.bitwise_and(img_sticky, left_leg)

cv.imwrite('sticky.jpg', img_sticky)

cv.imshow('Resultant Image', img_sticky)

cv.waitKey(0)
cv.destroyAllWindows()
