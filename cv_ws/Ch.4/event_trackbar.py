import numpy as np
import cv2

def onChange(value):
    global image, title

    add_value = value - int(image[0][0])
    print("추가 화솟값 : ", add_value)
    image = np.full((300,500), add_value, np.int8)
    cv2.imshow(title, image)

image = np.zeros((300,500), np.uint8)

title = 'Trackbar Event'
cv2.imshow(title, image)

cv2.createTrackbar('Brightness', title, image[0][0], 255, onChange)
cv2.waitKey(0)
cv2.destroyAllWindows()