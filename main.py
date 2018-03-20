import glob
import os
from random import shuffle
import cv2
import numpy as np
from scipy import linalg
from piro_lib import side_of_line, fit_division_line


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


print('OpenCV version: ' + cv2.__version__)

path = os.path.join('sets', '**', '*.png')

file_paths = glob.glob(path)
shuffle(file_paths)

for image_path in file_paths:

    img = cv2.imread(image_path)
    img2 = img.copy()
    img3 = img.copy()
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

    sides = sides >= 0
    sides = np.logical_xor(sides, np.roll(sides, -1))
    crosses = np.argwhere(sides).ravel()

    try:
        lengths = [(linalg.norm(contour_approx[ind, 0] - contour_approx[(ind+1)%contour_approx.shape[0], 0]), ind) for ind in crosses]
        longest_idx = crosses[lengths.index(max(lengths))]
        x1, y1 = contour_approx[longest_idx % contour_approx.shape[0], 0]
        x2, y2 = contour_approx[(longest_idx+1) % contour_approx.shape[0], 0]
        cv2.drawMarker(img2, (x1, y1), (255, 255, 255), cv2.MARKER_STAR)
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        print(e)

    # centroid
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawMarker(img3, (cX, cY), (255, 100, 0), cv2.MARKER_STAR)
    # corners
    corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in contour_approx:
        x, y = i.ravel()
        cv2.circle(img3, (x, y), 3, 255, -1)
    #

    fit_tuple = fit_division_line(cX, cY, contour_approx)
    if fit_tuple is not None:
        under, (vx, vy, x0, y0) = fit_tuple

        cv2.line(img3, (x0-vx*1000, y0-vy*1000), (x0+vx*1000, y0+vy*1000), (255, 255, 0))

        for point in under:
            x, y = point.ravel()
            cv2.drawMarker(img3, (x, y), (255, 100, 0), cv2.MARKER_TRIANGLE_DOWN)

    cv2.imshow('longest edge', img)
    cv2.imshow('regression', img2)
    cv2.imshow('centroid', img3)

    key = cv2.waitKey()
    if key == 27:  # ESC
        exit(0)
