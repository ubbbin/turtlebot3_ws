#import cv2

#image = cv2.imread('images/people.png')
#image_resized = cv2.resize(image, (755, 500))
#cascade_face_detector = cv2.CascadeClassifier('haarcascades//haarcascade_frontalface_default.xml')
#face_detections = cascade_face_detector.detectMultiScale(image_resized)
#for (x, y, w, h) in face_detections:
#    cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

#cv2.imshow("detect",image_resized)

#cv2.waitKey(0)

import cv2

image = cv2.imread('images/neo2.jpg',cv2.IMREAD_COLOR)
#image_resized = cv2.resize(image, (755, 500))
# cascade_face_detector = cv2.CascadeClassifier('haarcascades//haarcascade_frontalface_default.xml')
cascade_cat_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_detections = cascade_cat_detector.detectMultiScale(grayImg, scaleFactor=1.02, minNeighbors=2, minSize=(100,100))
for (x, y, w, h) in face_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("detect",image)

cv2.waitKey(0)