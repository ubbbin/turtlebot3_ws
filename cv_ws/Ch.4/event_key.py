import numpy as np
import cv2

switch_case = {
    ord('a') : "a키 입력",
    ord('b') : "b키 입력",
    0x41 : "A키 입력",
    2424832 : "왼쪽 화살표 키 입력",
    2490368 : "윗쪽 화살표 키 입력",
    2555904 : "오른쪽 화살표 키 입력",
    2621440 : "아래쪽 화살표 키 입력"
}

image = np.ones((200, 300), np.float32)
cv2.namedWindow('Keyboard Event')
cv2.imshow("Keyboard Event", image)

while True:
    key = cv2.waitKeyEx(100)
    if key == 27: break

    try:
        result = switch_case[key]
        print(result)

    except KeyError:
        result = -1

cv2.destroyAllWindows()