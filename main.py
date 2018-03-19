import os
import glob
from random import shuffle

import cv2
import numpy as np
from scipy import linalg


def calculate_edge_lengths(contour):
    diffs = contour - np.roll(contour, -1, axis=0)
    magnitudes = linalg.norm(diffs, axis=2).ravel()
    return magnitudes


def cut_polygon(polygon, start_point, end_point):
    start_index = np.where((polygon == start_point).all(axis=2))[0][0]
    end_index = np.where((polygon == end_point).all(axis=2))[0][0]

    if start_index < end_index:
        return polygon[start_index:end_index, :, :]
    else:
        return np.concatenate([polygon[start_index:, :, :], polygon[:end_index, :, :]], axis=0)


def side_of_line(points, line):
    vx, vy, x0, y0 = line

    # d = (x - x0) * vy - (y - y0) * vx

    points = (points - np.asarray([[[x0[0], y0[0]]]]))
    points *= np.asarray([[[vy[0], vx[0]]]])
    distances = np.diff(points, axis=2).ravel()

    # distances = [(x - x0)*vy - (y - y0)*vx for x, y in points[:, 0]]

    return distances

print('OpenCV version: ' + cv2.__version__)

path = os.path.join('sets', 'set8', '*.png')

file_paths = glob.glob(path)
shuffle(file_paths)

for image_path in file_paths:

    img = cv2.imread(image_path)
    img2 = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_approx = cv2.approxPolyDP(contours[0], 3.5, closed=True)

    edge_lengths = calculate_edge_lengths(contour_approx)

    longest_edge_index = np.argmax(edge_lengths)

    x1, y1 = contour_approx[longest_edge_index-1, 0]
    x2, y2 = contour_approx[(longest_edge_index+2) % len(contour_approx), 0]

    edge_of_cut = cut_polygon(contours[0], (x2, y2), (x1, y1))

    cv2.drawMarker(img, (x1, y1), (255, 100, 0), cv2.MARKER_CROSS)
    cv2.drawMarker(img, (x2, y2), (255, 100, 0), cv2.MARKER_STAR)

    cv2.drawContours(img, [contour_approx], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.polylines(img, [edge_of_cut], False, (255, 0, 0), 1, cv2.LINE_4)

    vx, vy, x0, y0 = cv2.fitLine(contour_approx, cv2.DIST_HUBER, 0, 0.01, 0.01)
    m = 1000
    cv2.line(img2, (x0 - m * vx, y0 - m * vy), (x0 + m * vx, y0 + m * vy), (100, 0, 100))
    cv2.line(img2, (x0 + m * vy, y0 - m * vx), (x0 - m * vy, y0 + m * vx), (255, 0, 255))

    sides = side_of_line(contour_approx, (vy, -vx, x0, y0))
    for side, pt in zip(sides, contour_approx):
        color = (0, 0, 255) if side > 0 else (0, 0, 150)
        x, y = pt[0]
        cv2.drawMarker(img2, (x, y), color, cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

    cv2.imshow('image', img)
    cv2.imshow('image2', img2)
    key = cv2.waitKey()
    if key == 27:  # ESC
        exit(0)
