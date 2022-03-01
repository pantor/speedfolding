import cv2 as cv
import numpy as np

from database import Database
from reward import segment 
from heuristics.planar_transform import PlanarTransform
from heuristics.utils import find_first_point_in_contour, get_normal_theta_at_contour_index, maximize_area_of_polygon_along_contour


class MoveHeuristic:
    def __init__(self):
        self.current_transform = None
        self.margin = {'top': 60, 'right': 10, 'bottom': 90, 'left': 110}

    def should_move_for_folding(self, image, center_threshold=0.4, top_threshold_px=50):
        _, info = segment(image)
        return info['rect'][1] < top_threshold_px and info['y'] < center_threshold

    def calculate(self, image, target_point=None, x_only=False, y_only=False, save=False, x_offset=-40, y_offset=90, margin_bottom=None):
        """moves to the center of the image in front of the robot if target_point is not given"""
        mask, info, contour = segment(image, return_contour=True)

        center_mask = np.array([info['x'] * image.shape[1], info['y'] * image.shape[0]])
        center_image = np.array([0.5 * image.shape[1] + y_offset, 0.5 * image.shape[0] + x_offset])  # Center before robot

        target_point = target_point if target_point is not None else center_image
        if x_only:
            target_point[0] = center_mask[0]
        elif y_only:
            target_point[1] = center_mask[1]

        theta = -np.arctan2(target_point[1] - center_mask[1], target_point[0] - center_mask[0])

        direction_left = center_mask + PlanarTransform.get_rotation_matrix(theta) @ np.array([0, 800])
        direction_right = center_mask + PlanarTransform.get_rotation_matrix(theta) @ np.array([0, -800])

        point_left = find_first_point_in_contour(direction_left, direction_right, contour)
        point_right = find_first_point_in_contour(direction_right, direction_left, contour)

        for i in range(20+1):
            new_direction = center_mask + PlanarTransform.get_rotation_matrix(theta - i*np.pi/20) @ np.array([0, 800])
            cv.line(mask, center_mask.astype(int), new_direction.astype(int), (0, 0, 0), 3)

        contours2, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        max_idx = np.argmax([cv.contourArea(c) for c in contours2])
        new_contour = contours2[max_idx]

        if margin_bottom is not None:
            self.margin['bottom'] = margin_bottom

        rect = (self.margin['left'], self.margin['top']), (image.shape[1] - self.margin['right'], image.shape[0] - self.margin['bottom'])
        mask_path = [(rect[0][0], rect[1][1]), rect[1], (rect[1][0], rect[0][1]), rect[0]]
        point_left_idx, point_right_idx = maximize_area_of_polygon_along_contour(new_contour, point_left, point_right, mask_path=mask_path)

        pick1 = PlanarTransform(
            position=new_contour[point_left_idx][0],
            theta=get_normal_theta_at_contour_index(point_left_idx, new_contour) + np.pi
        )
        pick2 = PlanarTransform(
            position=new_contour[point_right_idx][0],
            theta=get_normal_theta_at_contour_index(point_right_idx, new_contour) + np.pi
        )

        self.current_transform = target_point - center_mask

        place1 = PlanarTransform(position=pick1.position + self.current_transform, theta=pick1.theta)
        place2 = PlanarTransform(position=pick2.position + self.current_transform, theta=pick2.theta)

        if save:
            pick1.draw(image)
            pick2.draw(image)
            place1.draw(image)
            place2.draw(image)
            cv.imwrite('data/current/output.png', image)

        return pick1, pick2, place1, place2


if __name__ == '__main__':
    db = Database('test')
    # img = db.get_image('2021-12-20-18-17-24-090', 0, 'before', 'color')
    img = cv.imread('data/current/asdf.png')

    mh = MoveHeuristic()
    pick1, pick2, place1, place2 = mh.calculate(img, x_only=True, save=True, x_offset=0)
