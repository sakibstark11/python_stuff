import cv2
import numpy as np

def none(x):
    pass
cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame")
cv2.createTrackbar("L-H", "Frame", 0, 180, none)
cv2.createTrackbar("L-S", "Frame", 0, 255, none)
cv2.createTrackbar("L-V", "Frame", 0, 255, none)
cv2.createTrackbar("U-H", "Frame", 180, 180, none)
cv2.createTrackbar("U-S", "Frame", 255, 255, none)
cv2.createTrackbar("U-V", "Frame", 255, 255, none)
while True:
    _, frame = cap.read()
    lh = cv2.getTrackbarPos("L-H", "Frame")
    ls = cv2.getTrackbarPos("L-S", "Frame")
    lv = cv2.getTrackbarPos("L-V", "Frame")
    uh = cv2.getTrackbarPos("U-H", "Frame")
    us = cv2.getTrackbarPos("U-S", "Frame")
    uv = cv2.getTrackbarPos("U-V", "Frame")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([lh,ls,lv])
    upper_red = np.array([uh,us,uv])
    mask = cv2.inRange(hsv,lower_red, upper_red)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

