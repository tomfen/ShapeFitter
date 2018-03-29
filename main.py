import glob
import os
from random import shuffle
from piro_lib import *

print('OpenCV version: ' + cv2.__version__)

path = os.path.join('sets', '**', '*.png')

file_paths = glob.glob(path)
shuffle(file_paths)

for image_path in file_paths:

    print(image_path)

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_approx = cv2.approxPolyDP(contours[0], 3.75, closed=True)

    cX, cY = centroid(contours[0])

    vx, vy, x0, y0 = cv2.fitLine(contour_approx, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    m = 1000
    cv2.line(img, (int(x0 - m * vx), int(y0 - m * vy)), (int(x0 + m * vx), int(y0 + m * vy)), (100, 0, 100))
    cv2.line(img, (int(cX + m * vy), int(cY - m * vx)), (int(cX - m * vy), int(cY + m * vx)), (255, 0, 255))
    cv2.circle(img, (int(cX), int(cY)), 3, (50, 50, 255), 3)

    sides = side_of_line(contour_approx, (vy, -vx, cX, cY))
    for side, pt in zip(sides, contour_approx):
        color = (0, 0, 255) if side > 0 else (0, 0, 150)
        x, y = pt[0]
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

    sides = np.greater_equal(sides, 0)
    sides = np.logical_xor(sides, np.roll(sides, -1))
    crosses = np.argwhere(sides).ravel()

    try:
        lengths = [(np.linalg.norm(contour_approx[ind, 0] - contour_approx[(ind+1) % contour_approx.shape[0], 0]), ind)
                   for ind in crosses]
        longest_idx = crosses[lengths.index(max(lengths))]
        x1, y1 = contour_approx[longest_idx % contour_approx.shape[0], 0]
        x2, y2 = contour_approx[(longest_idx+1) % contour_approx.shape[0], 0]
        cv2.drawMarker(img, (x1, y1), (255, 255, 255), cv2.MARKER_STAR)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        print(e)

    cv2.imshow('regression', img)

    key = cv2.waitKey()
    if key == 27:  # ESC
        exit(0)
