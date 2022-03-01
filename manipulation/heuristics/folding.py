import cv2 as cv
import numpy as np

from heuristics.planar_transform import PlanarTransform
from heuristics.utils import find_first_point_in_contour, get_normal_theta_at_contour_index, maximize_area_of_polygon_along_contour
from reward import segment 


class Line:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

    def to_absolute_pixels(self, image_shape):
        return Line(
            start=[image_shape[1]*self.start[0], image_shape[0]*self.start[1]],
            end=[image_shape[1]*self.end[0], image_shape[0]*self.end[1]]
        )


class FoldingHeuristic:
    def __init__(self):
        self.mask_path = [(110, 84), (1000, 104), (1000, 610), (110, 560)]

    @staticmethod
    def getPerpCoord(a, b, length):
        v = (b - a) / np.linalg.norm(b - a)
        temp = v[0]
        v[0] = 0 - v[1]
        v[1] = temp
        return b + v * length

    @staticmethod
    def mirrorPoint(a, b, p):
        diff = np.array([a[1] - b[1], b[0] - a[0]])
        D = 2 * (diff[0] * (p[0] - a[0]) + diff[1] * (p[1] - a[1])) / np.linalg.norm(diff)**2
        return np.array([p[0] - diff[0] * D, p[1] - diff[1] * D])

    def calculate(self, image, line, same_gripper_distance=30.0, save=False):
        new_line = line.to_absolute_pixels(image.shape)

        img_thresh, _, all_contour = segment(image, return_contour=True)

        c = self.getPerpCoord(new_line.start, new_line.end, 800)
        d = self.getPerpCoord(new_line.end, new_line.start, -800)

        pts = np.array([[new_line.start, new_line.end, c, d]])
        cv.fillPoly(img_thresh, pts.astype(int), 0)
        contours2, _ = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        areas = [cv.contourArea(c) for c in contours2]
        second_max_index = np.argsort(areas, axis=0)[-1]
        secondary_contour = contours2[second_max_index]

        c1 = find_first_point_in_contour(new_line.start, new_line.end, all_contour)
        c2 = find_first_point_in_contour(new_line.end, new_line.start, all_contour)

        # cv.drawContours(image, [all_contour], 0, (0, 255, 0), 1)
        # cv.circle(image, c1.astype(int), 4, (255, 255, 255), -1)
        # cv.circle(image, c2.astype(int), 4, (255, 255, 255), -1)

        mask_mirrored_path = [tuple(self.mirrorPoint(new_line.start, new_line.end, p).astype(int)) for p in self.mask_path]
        (point_left_idx, point_right_idx) = maximize_area_of_polygon_along_contour(secondary_contour, c1, c2, mask_path=(self.mask_path, mask_mirrored_path))

        pick_left = PlanarTransform(
            position=secondary_contour[point_left_idx][0],
            theta=get_normal_theta_at_contour_index(point_left_idx, secondary_contour),
        )

        pick_right = PlanarTransform(
            position=secondary_contour[point_right_idx][0],
            theta=get_normal_theta_at_contour_index(point_right_idx, secondary_contour),
        )

        place_left = pick_left.mirror_along_line(new_line)
        place_right = pick_right.mirror_along_line(new_line)

        # Currently set place orientation to pick orientation
        place_left.theta = pick_left.theta
        place_right.theta = pick_right.theta
        
        # Check if two points are nearby
        if np.linalg.norm(pick_left.position - pick_right.position, 2) < same_gripper_distance:
            pick_left.position = (pick_left.position + pick_right.position) / 2
            pick_left.theta = (pick_left.theta + pick_right.theta) / 2
            pick_right = None
            place_right = None

        if save:
            img = np.copy(image)
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            
            self.draw_mask(img)
            self.draw_plan(img, line, (pick_left, pick_right, place_left, place_right))
            cv.imwrite('data/current/output.png', img)
        
        return pick_left, pick_right, place_left, place_right

    def draw_mask(self, image):
        pts = np.array(self.mask_path, np.int32).reshape((1, -1, 2))
        cv.polylines(image, pts, True, (0, 255, 255), 1)

    def draw_plan(self, image, line, points):
        new_line = line.to_absolute_pixels(image.shape)

        pick_left, pick_right, place_left, place_right = points

        cv.line(image, new_line.start.astype(int), new_line.end.astype(int), (0, 120, 255), 3)

        # Draw pick and place points
        pick_left.draw(image)
        place_left.draw(image)
        
        if pick_right is not None:
            pick_right.draw(image)
            place_right.draw(image)


if __name__ == '__main__':
    from database import Database

    db = Database(collection='test')
    img = db.get_image('2022-01-21-12-16-01-910', 0, 'before', 'color')
    # img = cv.imread('data/current/image-color.png')
    if img is None:
        raise Exception('Image not found!')

    line = Line(start=[0.99, 0.44], end=[0.01, 0.37])

    fh = FoldingHeuristic()
    pick_left, pick_right, place_left, place_right = fh.calculate(img, line, save=True)
