import math
import numpy as np


def side_of_line(points, line):
    vx, vy, x0, y0 = line

    # d = (x - x0) * vy - (y - y0) * vx

    points = (points - np.asarray([[[x0[0], y0[0]]]]))
    points *= np.asarray([[[vy[0], vx[0]]]])
    distances = np.diff(points, axis=2).ravel()

    # distances = [(x - x0)*vy - (y - y0)*vx for x, y in points[:, 0]]

    return distances


def fit_division_line(x, y, corners):

    for angle in range(0, 360, 1):

        angle = math.pi*2 * angle / 360

        vx = math.cos(angle)
        vy = math.sin(angle)

        line = np.asarray([[vx], [vy], [x], [y]])

        sides = side_of_line(corners, line) > 0

        if np.count_nonzero(sides) == 2:
            under = np.compress(sides, corners, axis=0)
            return under, line

    return None
