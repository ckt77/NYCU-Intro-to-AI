import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')
ret, background = cap.read()
height, width, channels = background.shape

frame_counter = 0
while True:
    frame_counter += 1
    print(frame_counter)

    ret, frame = cap.read()
    if not ret or frame_counter == 130:
        foreground = np.zeros((height, width, channels))
        foreground[:, :, 1] = thresh
        break

    diff = cv2.absdiff(background, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    background = frame


result = np.hstack((background, foreground))
cv2.imwrite('hw0_109652025_2.png', result)

cap.release()
cv2.destroyAllWindows()
