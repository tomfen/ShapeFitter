import os
import glob
import cv2
import numpy as np
from scipy import linalg


def calculate_edge_lengths(contour):
    a = contour[:]
    b = np.roll(a, -1, axis=0)
    c = a - b
    d = linalg.norm(c, axis=2).ravel()
    return d


print('OpenCV version: ' + cv2.__version__)

path = os.path.join('sets', 'set8', '*.png')

for image_path in glob.iglob(path):

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_approx = cv2.approxPolyDP(contours[0], 5, closed=True)

    cv2.drawContours(img, contours,          -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.drawContours(img, [contour_approx], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    edge_lengths = calculate_edge_lengths(contour_approx)

    longest_edge_index = np.argmax(edge_lengths)

    x1, y1 = contour_approx[longest_edge_index, 0]
    x2, y2 = contour_approx[(longest_edge_index+1) % len(contour_approx), 0]

    cv2.drawMarker(img, (x1, y1), (255, 100, 0), cv2.MARKER_CROSS)
    cv2.drawMarker(img, (x2, y2), (255, 100, 0), cv2.MARKER_CROSS)

    cv2.imshow('image', img)
    key = cv2.waitKey()
    if key == 27:  # ESC
        exit(0)



