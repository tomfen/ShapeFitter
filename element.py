import cv2
from piro_lib import centroid, distances_from_line, cut_polygon
import numpy as np


class Element:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour = contours[0]
        self.contour_approx = cv2.approxPolyDP(self.contour, 3.75, closed=True)

        self.centroid = (cx, cy) = centroid(self.contour)
        self.regression_line = (vx, vy, x0, y0) = cv2.fitLine(self.contour_approx, cv2.DIST_HUBER, 0, .01, .01).ravel()

        self.approx_cont_line_distances = distances_from_line(self.contour_approx, (vy, -vx, cx, cy))

        sides = np.greater_equal(self.approx_cont_line_distances, 0)
        side_changes = np.logical_xor(sides, np.roll(sides, -1))
        side_changes_indices = np.argwhere(side_changes).ravel()

        lengths = self.edge_lengths(self.contour_approx, side_changes_indices)
        longest_idx = side_changes_indices[lengths.index(max(lengths))]

        self.cut_end = self.contour_approx[(longest_idx - 1) % self.contour_approx.shape[0], 0]
        self.cut_start = self.contour_approx[(longest_idx + 2) % self.contour_approx.shape[0], 0]

        self.cut = cut_polygon(self.contour, self.cut_start, self.cut_end)

        self.cut_normalized = self.normalize(self.cut)

    def representation(self):
        img_copy = self.img.copy()

        m = 1000

        vx, vy, x0, y0 = self.regression_line
        cx, cy = self.centroid

        cv2.line(img_copy, (int(x0 - m * vx), int(y0 - m * vy)), (int(x0 + m * vx), int(y0 + m * vy)), (100, 0, 100))
        cv2.line(img_copy, (int(cx + m * vy), int(cy - m * vx)), (int(cx - m * vy), int(cy + m * vx)), (255, 0, 255))
        cv2.circle(img_copy, (int(cx), int(cy)), 3, (255, 100, 0), 2)

        boundary = cut_polygon(self.contour, self.cut_end, self.cut_start)

        cv2.polylines(img_copy, [boundary], False, (255, 0, 0), 2)
        cv2.polylines(img_copy, [self.cut], False, (0, 255, 0), 2)

        for side, pt in zip(self.approx_cont_line_distances, self.contour_approx):
            color = (0, 0, 255) if side > 0 else (0, 0, 150)
            x, cy = pt[0]
            cv2.drawMarker(img_copy, (x, cy), color, cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

        return img_copy

    @staticmethod
    def edge_lengths(contour, indices):

        lengths = []

        for i in indices:
            p1 = contour[i, 0]
            p2 = contour[(i + 1) % contour.shape[0], 0]
            distance = np.linalg.norm(p1-p2)

            lengths.append(distance)

        return lengths

    @staticmethod
    def similarity(element1, element2):
        # TODO
        return np.random.random()

    @staticmethod
    def normalize(cut):
        ret = cut.copy()
        center = np.floor_divide((ret[-1:, :, :] + ret[:1, :, :]), 2)
        ret -= center
        ret = ret.astype("float")

        _x, _y = ret[0, 0]
        a = np.math.atan2(_y, _x)
        sin = np.math.sin(a)
        cos = np.math.cos(a)

        for i in range(ret.shape[0]):
            pt = ret[i, 0]
            pt = [sin * pt[0] - cos * pt[1], cos * pt[0] + sin * pt[1]]
            ret[i, 0] = pt

        return ret

    def cut_representation(self):
        img = np.zeros_like(self.img)
        shift = [[[int(img.shape[1]/2), int(img.shape[0]/2)]]]
        cut = self.cut_normalized.astype("int")
        cut += shift

        cv2.line(img, (int(img.shape[1]/2), 0), (int(img.shape[1]/2), img.shape[0]), (50, 50, 50), 1)

        cv2.polylines(img, [cut], False, (255, 255, 255), 1)
        cv2.circle(img, tuple(shift[0][0]), 4, (255, 255, 255), 1)
        return img
