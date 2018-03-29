import cv2
import numpy as np


def side_of_line(points, line):
    vx, vy, x0, y0 = line
    # d = (x - x0) * vy - (y - y0) * vx

    points = (points - np.asarray([[[x0, y0]]]))
    points *= np.asarray([[[vy, vx]]])
    distances = np.diff(points, axis=2).ravel()

    return distances


def centroid(contour):
    m = cv2.moments(contour)
    x = m["m10"] / m["m00"]
    y = m["m01"] / m["m00"]
    return x, y


def cut_polygon(polygon, start_point, end_point):
    start_index = np.where(np.equal(polygon, start_point).all(axis=2))[0][0]
    end_index = np.where(np.equal(polygon, end_point).all(axis=2))[0][0]
    if start_index < end_index:
        return polygon[start_index:end_index, :, :]
    else:
        return np.concatenate([polygon[start_index:, :, :], polygon[:end_index, :, :]], axis=0)


def calculate_edge_lengths(contour):
    diffs = contour - np.roll(contour, -1, axis=0)
    magnitudes = np.linalg.norm(diffs, axis=2).ravel()
    return magnitudes
