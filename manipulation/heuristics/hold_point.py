import time

import cv2 as cv
from loguru import logger
import numpy as np

from drawing import draw_mask
from reward import segment
from heuristics.planar_transform import PlanarTransform
from heuristics.utils import find_first_point_in_contour


class HoldPointHeuristic:
    def __init__(self):
        self.margin = {'top': 162, 'right': 20, 'bottom': 100, 'left': 140}
        self.return_none_if_outside_mask = True

    def refine_place_and_hold_points(self, pick: PlanarTransform, place: PlanarTransform, image, min_distance_px=110):
        _, _, contour = segment(image, return_contour=True)
        contour = np.squeeze(contour)

        unit_pick_to_place = (place.position - pick.position) / np.linalg.norm(place.position - pick.position)
        left = pick.position + 1000 * unit_pick_to_place
        right = place.position - 1000 * unit_pick_to_place
        near_pick = find_first_point_in_contour(left, right, contour)
        far_pick = find_first_point_in_contour(right, left, contour)

        def get_dist(pos):
            near_dist = (pick.position - pos) / unit_pick_to_place
            near_dist[np.isinf(near_dist)] = np.nan
            return np.nanmean(near_dist)

        near_dist, far_dist = get_dist(near_pick), get_dist(far_pick)

        if far_dist < min_distance_px:
            return place, None

        hold_dist = np.clip(0.6 * near_dist + 0.4 * far_dist, min_distance_px, far_dist + 3.0)
        hold = pick.position - hold_dist * unit_pick_to_place

        # Check if hold point is outside mask/margin
        if self.return_none_if_outside_mask:
            mask_tl = np.array([self.margin['left'], self.margin['top']])
            mask_br = image.shape[1::-1] - np.array([self.margin['right'], self.margin['bottom']])

            if not (mask_tl[0] <= hold[0] <= mask_br[0] and mask_tl[1] <= hold[1] <= mask_br[1]):
                logger.warning('hold point is outside mask, so return none')
                return place, None

        hold_pixel = PlanarTransform(position=hold, theta=np.arctan2(unit_pick_to_place[0], unit_pick_to_place[1]) + np.pi/2)
        return place, hold_pixel


if __name__ == '__main__':
    from database import Database

    db = Database(collection='test')

    hph = HoldPointHeuristic()

    for action in db.yield_actions(is_self_supervised=True, primitive_type='pick-and-hold'):
        image = db.get_image(action['episode_id'], action['action_id'], 'before', 'color')
        pixel1 = PlanarTransform.fromRelativePixelDictionary(action['poses'][0], image.shape)
        pixel2 = PlanarTransform.fromRelativePixelDictionary(action['poses'][1], image.shape)
        refined_place, hold = hph.refine_place_and_hold_points(pixel1, pixel2, image)

        pixel1.draw(image)
        if hold is not None:
            hold.draw(image, color=(0, 255, 255))

        refined_place.draw(image, color=(0, 0, 255))
        draw_mask(image, hph.margin)

        cv.imwrite('data/current/plan.png', image)
        break
        time.sleep(0.8)
