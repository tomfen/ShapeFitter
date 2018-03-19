import os
import glob
import cv2
import numpy as np
from scipy import linalg

from piro_lib import fit_division_line

def calculate_edge_lengths(contour):
    a = contour[:]
    b = np.roll(a, -1, axis=0)
    c = a - b
    d = linalg.norm(c, axis=2).ravel()
    return d

def cut_edge(edge, start, end):
    start_index = np.where((edge == start).all(axis=2))[0][0]
    end_index = np.where((edge == end).all(axis=2))[0][0]

    if start_index < end_index:
        return edge[start_index:end_index, :, :]
    else:
        return np.concatenate([edge[start_index:, :, :], edge[:end_index, :, :]], axis=0)

print('OpenCV version: ' + cv2.__version__)

path = os.path.join('sets', '**', '*.png')

for image_path in glob.iglob(path):

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_approx = cv2.approxPolyDP(contours[0], 4, closed=True)
    edge_lengths = calculate_edge_lengths(contour_approx)
    longest_edge_index = np.argmax(edge_lengths)


    x1, y1 = contour_approx[longest_edge_index-1, 0]
    x2, y2 = contour_approx[(longest_edge_index+2) % len(contour_approx), 0]

    edge = cut_edge(contours[0], (x2, y2), (x1, y1))

    cv2.drawMarker(img, (x1, y1), (255, 100, 0), cv2.MARKER_CROSS)
    cv2.drawMarker(img, (x2, y2), (255, 100, 0), cv2.MARKER_STAR)

    cv2.drawContours(img, [contour_approx], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.polylines(img, [edge], False, (255, 0, 0), 1, cv2.LINE_4)

    vx, vy, x0, y0 = cv2.fitLine(contour_approx, cv2.DIST_L2, 0, 0.01, 0.01)
    m = 1000
    cv2.line(img, (x0 - m * vx, y0 - m * vy), (x0 + m * vx, y0 + m * vy), (255, 0, 255))
    cv2.line(img, (x0, y0), (x0 - m * vy, y0 + m * vx), (255, 0, 255))

    # centroid
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawMarker(img, (cX, cY), (255, 100, 0), cv2.MARKER_STAR)
    # corners
    corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    #

    fit_tuple = fit_division_line(cX, cY, corners)
    if fit_tuple is not None:
        a, b, under = fit_tuple
        max_x = img.shape[0]
        cv2.line(img, (0, int(b)), (max_x, int(max_x * a + b)), (255, 255, 0))
        for u in under:
            u_x, u_y = u.ravel()
            cv2.drawMarker(img, (u_x, u_y), (255, 255, 0), cv2.MARKER_TILTED_CROSS)

    cv2.imshow(image_path, img)

    key = cv2.waitKey()
    if key == 27:  # ESC
        exit(0)



