import cv2
from cv2 import threshold
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret is False:
        break

    # roi = frame[269: 795, 537: 1416]
    roi = frame

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    rows, cols, _ = roi.shape

    _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)

    if contours == []:
        print("Closed")
    
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)

        cv2.rectangle(roi, (x,y), (x+w, y+h), (250, 0, 0))
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)

        break

    cv2.imshow("Roi", gray_roi)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Roi", roi)

    key = cv2.waitKey(30)

    while True:
        if key == 32:
            key = cv2.waitKey(100)
        else:
            break

    if key == 27:
        break

cv2.destroyAllWindows()