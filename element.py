import cv2
import sys

from piro_lib import *
import numpy as np
from scipy import interpolate, signal


class Element:
    _SIGNAL_LENGTH = 300
    _SIGNAL_MAGNITUDE = 64
    _NORMALIZATION_HEIGHT = 128
    _SIDE_MIN_LENGTH = 20
    _GRAY_THRESHOLD = 127
    _ANGLE_TOLERANCE = 22.5

    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, img_gray = cv2.threshold(img_gray, Element._GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)

        _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour = max(contours, key=lambda i: len(i))
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

        self.pt_A = self.contour_approx[longest_idx, 0]
        self.pt_B = self.contour_approx[(longest_idx + 1) % self.contour_approx.shape[0], 0]

        # fixing cut points, if they're in wrong place
        side_length_left = distance(self.pt_A, self.cut_end)
        side_length_right = distance(self.pt_B, self.cut_start)

        if side_length_left < Element._SIDE_MIN_LENGTH:
            self.cut_end = self.contour_approx[(longest_idx - 2) % self.contour_approx.shape[0], 0]

        if side_length_right < Element._SIDE_MIN_LENGTH:
            self.cut_start = self.contour_approx[(longest_idx + 3) % self.contour_approx.shape[0], 0]

        self.validate_base_angles(longest_idx)

        self.pt_C, self.pt_D = self.find_cd()

        self.cut = cut_polygon(self.contour, self.cut_start, self.cut_end)

        self.cut_normalized = self.normalize()
        self.signal = self.as_signal(self.cut_normalized)

    def validate_base_angles(self, longest_idx):
        angle_left = angle(self.cut_end, self.pt_A, self.pt_B)
        angle_right = angle(self.pt_A, self.pt_B, self.cut_start)

        angle_min = 90 - Element._ANGLE_TOLERANCE
        angle_max = 90 + Element._ANGLE_TOLERANCE

        if angle_left < angle_min or angle_max < angle_left:
            self.cut_end = self.contour_approx[(longest_idx - 0) % self.contour_approx.shape[0], 0]
            self.pt_A = self.contour_approx[(longest_idx + 1) % self.contour_approx.shape[0], 0]
            self.pt_B = self.contour_approx[(longest_idx + 2) % self.contour_approx.shape[0], 0]
            self.cut_start = self.contour_approx[(longest_idx + 3) % self.contour_approx.shape[0], 0]

        if angle_right < angle_min or angle_max < angle_right:
            self.cut_end = self.contour_approx[(longest_idx - 2) % self.contour_approx.shape[0], 0]
            self.pt_A = self.contour_approx[(longest_idx - 1) % self.contour_approx.shape[0], 0]
            self.pt_B = self.contour_approx[(longest_idx + 0) % self.contour_approx.shape[0], 0]
            self.cut_start = self.contour_approx[(longest_idx + 1) % self.contour_approx.shape[0], 0]

    def find_cd(self):

        distance_ae = distance(self.pt_A, self.cut_end)
        distance_bs = distance(self.pt_B, self.cut_start)

        if distance_ae > distance_bs:
            c = self.pt_A + (self.cut_end - self.pt_A)*(distance_bs/distance_ae)
            return c, self.cut_start
        else:
            d = self.pt_B + (self.cut_start - self.pt_B)*(distance_ae/distance_bs)
            return self.cut_end, d

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

        cv2.drawMarker(img_copy, tuple(self.pt_A), (150, 150, 150), cv2.MARKER_CROSS, markerSize=10)
        cv2.putText(img_copy, 'A', tuple(self.pt_A), cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150))
        cv2.drawMarker(img_copy, tuple(self.pt_B), (150, 150, 150), cv2.MARKER_CROSS, markerSize=10)
        cv2.putText(img_copy, 'B', tuple(self.pt_B), cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150))
        cv2.drawMarker(img_copy, tuple(self.pt_C.astype('int')), (150, 150, 150), cv2.MARKER_CROSS, markerSize=10)
        cv2.putText(img_copy, 'C', tuple(self.pt_C.astype('int')), cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150))
        cv2.drawMarker(img_copy, tuple(self.pt_D.astype('int')), (150, 150, 150), cv2.MARKER_CROSS, markerSize=10)
        cv2.putText(img_copy, 'D', tuple(self.pt_D.astype('int')), cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150))

        return img_copy

    @staticmethod
    def edge_lengths(contour, indices):

        lengths = []

        for i in indices:
            p1 = contour[i, 0]
            p2 = contour[(i + 1) % contour.shape[0], 0]
            lengths.append(distance(p1, p2))

        return lengths

    @staticmethod
    def similarity(element1, element2):
        if len(element1.contour_approx) == 4 and len(element2.contour_approx) == 4:
            return float('inf')

        curve1 = element1.signal
        curve2 = element2.signal
        cross_correlation = signal.correlate(curve1, np.flip(-curve2, axis=0))
        return cross_correlation.max()

    def normalize(self):

        src_pts = np.asarray([[self.pt_A], [self.pt_B], [self.pt_C], [self.pt_D]])
        dst_pts = np.asarray([[[0, Element._NORMALIZATION_HEIGHT]],
                              [[Element._SIGNAL_LENGTH, Element._NORMALIZATION_HEIGHT]],
                              [[0, 0]],
                              [[Element._SIGNAL_LENGTH, 0]]])
        m, _ = cv2.findHomography(src_pts, dst_pts)

        ret = self.cut.copy()
        ret = ret.astype('float')

        for i in range(ret.shape[0]):
            pt = ret[i, 0]
            pt = [pt[0] * m[0, 0] + pt[1] * m[0, 1] + m[0, 2], pt[0] * m[1, 0] + pt[1] * m[1, 1] + m[1, 2]]
            ret[i, 0] = pt

        return ret

    @staticmethod
    def as_signal(curve):
        f = interpolate.interp1d(curve[:, 0, 0], curve[:, 0, 1], bounds_error=False, fill_value=0)
        signal = np.asarray([f(x) for x in range(Element._SIGNAL_LENGTH)])

        avg = np.average(signal)
        std = np.std(signal)

        if std == 0:
            std = 1

        return (signal - avg) / std

    @staticmethod
    def cut_representation(element1, element2):

        img_shape = (Element._SIGNAL_MAGNITUDE*5, Element._SIGNAL_LENGTH, 3)

        img = np.zeros(img_shape, np.uint8)
        shift = [[[0, img_shape[0]//2]]]
        cv2.line(img, (0, int(img.shape[0] / 2)), (img.shape[1], int(img.shape[0] / 2)), (50, 50, 50), 1)

        as_signal1 = np.asarray([[[x, y]] for x, y in enumerate(-np.flip(element1.signal, 0))])
        as_signal2 = np.asarray([[[x, y]] for x, y in enumerate(element2.signal)])

        as_signal1 = as_signal1 * np.asarray([[[1, Element._SIGNAL_MAGNITUDE]]]) + shift
        as_signal2 = as_signal2 * np.asarray([[[1, Element._SIGNAL_MAGNITUDE]]]) + shift

        cv2.polylines(img, [as_signal1.astype('int')], False, (255, 0, 255), 1)
        cv2.polylines(img, [as_signal2.astype('int')], False, (255, 255, 0), 1)

        return img
