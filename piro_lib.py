import cv2
import numpy as np


def distances_from_line(points, line):
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
        return polygon[start_index:end_index+1, :, :]
    else:
        return np.concatenate([polygon[start_index:, :, :], polygon[:end_index+1, :, :]], axis=0)


def edge_lengths(contour):
    diffs = contour - np.roll(contour, -1, axis=0)
    magnitudes = np.linalg.norm(diffs, axis=2).ravel()
    return magnitudes


def distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)


def angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.arccos(cosine_angle)

    return np.degrees(ang)
