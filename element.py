import cv2
from piro_lib import centroid, distances_from_line
import numpy as np


class Element:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour = contours[0]
        self.contour_approx = cv2.approxPolyDP(self.contour, 3.75, closed=True)

        self.centroid = (cx, cy) = centroid(self.contour)
        self.regression_line = (vx, vy, x0, y0) =\
            cv2.fitLine(self.contour_approx, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()

        self.approx_cont_line_distances = distances_from_line(self.contour_approx, (vy, -vx, cx, cy))

        sides = np.greater_equal(self.approx_cont_line_distances, 0)
        side_changes = np.logical_xor(sides, np.roll(sides, -1))
        side_changes = np.argwhere(side_changes).ravel()

        lengths = [
            (np.linalg.norm(self.contour_approx[ind, 0] - self.contour_approx[(ind + 1) % self.contour_approx.shape[0],
                                                                              0]), ind)
            for ind in side_changes]
        longest_idx = side_changes[lengths.index(max(lengths))]
        self.base_a = tuple(self.contour_approx[longest_idx % self.contour_approx.shape[0], 0])
        self.base_b = tuple(self.contour_approx[(longest_idx + 1) % self.contour_approx.shape[0], 0])

    def representation(self):
        img_copy = self.img.copy()

        m = 1000

        vx, vy, x0, y0 = self.regression_line
        cx, cy = self.centroid

        cv2.line(img_copy, (int(x0 - m * vx), int(y0 - m * vy)), (int(x0 + m * vx), int(y0 + m * vy)), (100, 0, 100))
        cv2.line(img_copy, (int(cx + m * vy), int(cy - m * vx)), (int(cx - m * vy), int(cy + m * vx)), (255, 0, 255))
        cv2.circle(img_copy, (int(cx), int(cy)), 3, (50, 50, 255), 3)
        
        cv2.drawMarker(img_copy, self.base_a, (255, 255, 255), cv2.MARKER_STAR)
        cv2.line(img_copy, self.base_a, self.base_b, (0, 0, 255), 2)

        for side, pt in zip(self.approx_cont_line_distances, self.contour_approx):
            color = (0, 0, 255) if side > 0 else (0, 0, 150)
            x, cy = pt[0]
            cv2.drawMarker(img_copy, (x, cy), color, cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

        return img_copy
