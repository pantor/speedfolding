import json
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

from reward import segment
from heuristics.planar_transform import PlanarTransform
from heuristics.utils import get_normal_theta_at_contour_index


class Instruction:
    instructions_path = Path('data') / 'instructions'

    def __init__(self, folding_lines, grasp_points=None):
        self.folding_lines = folding_lines
        self.grasp_points = grasp_points

    @classmethod
    def load_instructions(cls, name: str) -> List:
        with open(str(cls.instructions_path / f'{name}.json')) as f:
            data = json.load(f)
            return [Instruction(instr['folding-lines'], instr['grasp-points']) for instr in data['instructions']]

    @classmethod
    def get_template(cls, name: str):
        return cv.imread(str(cls.instructions_path / f'{name}-template.png'), cv.IMREAD_GRAYSCALE)

    def draw(self, image):
        for i, (start, end) in enumerate(self.folding_lines):
            l1 = np.array(start)
            l2 = np.array(end)
            cv.line(image, l1.astype(int), l2.astype(int), (255, 255, 0), 2)
            cv.putText(image, f'{i}', (l1 + [2, -10]).astype(int), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 120), 2, cv.LINE_AA)

        if self.grasp_points is not None:
            cv.circle(image, np.array(self.grasp_points[0]).astype(int), 2, (0, 0, 0), 2)
            cv.circle(image, np.array(self.grasp_points[1]).astype(int), 2, (0, 0, 0), 2)


class TemplateMatching:
    def __init__(self):
        self.rng = np.random.default_rng()

    @staticmethod
    def rotate_image(image, theta: float, offset=(0, 0)):
        center = np.array(image.shape[1::-1]) / 2
        t = cv.getRotationMatrix2D(center, -theta * 180 / np.pi, 1.0)
        t[0, 2] += offset[0]
        t[1, 2] += offset[1]
        return cv.warpAffine(image, t, image.shape[1::-1])

    def get_matched_instruction(self, mask, template_name: str, save=False, image=None):
        # Make sure that the center of mass of the template's segmentation mask is precisely at the center of the image        
        template = Instruction.get_template(template_name)

        mask_color = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        template_color = cv.cvtColor(template, cv.COLOR_GRAY2RGB)

        # Initital guess based on mask
        _, mask_info = segment(mask_color)
        _, template_info = segment(template_color)

        x = (mask_info['x'] - template_info['x']) * template.shape[1]
        y = (mask_info['y'] - template_info['y']) * template.shape[0]

        def score_pose(pose):
            return np.mean((mask == self.rotate_image(template, pose[2], pose[:2])).astype(np.float32))

        # Particle based template matching
        best_pose, best_score = [], 0.0
        for theta in np.linspace(0.0, 2*np.pi, 60):
            new_pose = np.array([x, y, theta])
            score = score_pose(new_pose)
            if score > best_score:
                best_score = score
                best_pose = new_pose

        for i in range(12):
            std = np.array([40.0, 40.0, 0.4]) / (i + 1)
            new_poses = self.rng.normal(best_pose, std, size=(20, 3))
            for new_pose in new_poses:
                score = score_pose(new_pose)
                if score > best_score:
                    best_score = score
                    best_pose = new_pose

        # Transform lines into new frame
        center = np.array(mask.shape[1::-1]) / 2
        offset = best_pose[:2] + center
        rot_mat = PlanarTransform.get_rotation_matrix(-best_pose[2])

        instructions = Instruction.load_instructions(template_name)

        for instr in instructions:
            for j in range(len(instr.folding_lines)):
                instr.folding_lines[j][0] = rot_mat @ (instr.folding_lines[j][0] - center) + offset
                instr.folding_lines[j][1] = rot_mat @ (instr.folding_lines[j][1] - center) + offset

            if instr.grasp_points is not None:
                instr.grasp_points[0] = rot_mat @ (instr.grasp_points[0] - center) + offset
                instr.grasp_points[1] = rot_mat @ (instr.grasp_points[1] - center) + offset

        # Rank possible instructions according to y position of first line
        instruction = sorted(instructions, key=lambda x: x.folding_lines[0][0][1] + x.folding_lines[0][1][1], reverse=True)[0]

        if save:
            image_draw = image if image is not None else mask
            image_draw = cv.cvtColor(image_draw, cv.COLOR_GRAY2RGB)

            if image is not None:
                fitted_mask = self.rotate_image(template, best_pose[2], best_pose[:2])
                image_draw[:, :, 2] = cv.addWeighted(image_draw[:, :, 2], 0.75, mask, 0.25, 0)
                image_draw[:, :, 1] = cv.addWeighted(image_draw[:, :, 1], 0.75, fitted_mask, 0.25, 0)
                
            instruction.draw(image_draw)
            cv.imwrite('data/current/output.png', image_draw)

        return instruction

    def find_nearest_grasp_pose(self, image, grasp_point, save=False) -> PlanarTransform:
        _, _, contour = segment(image, return_contour=True)

        min_idx = np.argmin(np.linalg.norm(contour - grasp_point, axis=2))

        # Interpolate the contour in a neighborhood around the nearest point
        a = range(min_idx-3, min_idx+4)
        part = contour.take(a, axis=0, mode='wrap')

        # np.interp only takes list of scalars
        inter_x = np.interp(np.linspace(a[0], a[-1], num=64), a, part[:, 0, 0])
        inter_y = np.interp(np.linspace(a[0], a[-1], num=64), a, part[:, 0, 1])

        inter_part = np.expand_dims(np.stack((inter_x, inter_y), axis=1), 1)
        min_idx = np.argmin(np.linalg.norm(inter_part - grasp_point, axis=2))

        pick = PlanarTransform(
            position=inter_part[min_idx][0],
            theta=get_normal_theta_at_contour_index(min_idx, inter_part),
        )

        if save:
            cv.circle(image, grasp_point, 3, (255, 0, 0))
            pick.draw(image)
            cv.imwrite('data/current/output.png', image)

        return pick


if __name__ == '__main__':
    from database import Database

    db = Database(collection='test')
    image = db.get_image('2021-12-28-11-48-12-651', 0, 'before', 'color')
    # image = cv.imread('data/current/image-color.png')
    # image = cv.imread('data/instructions/towel-template.png')
    mask, _ = segment(image)

    m = TemplateMatching()

    instruction = m.get_matched_instruction(mask, template_name='shirt', save=True, image=image)
    
    # m.find_nearest_grasp_pose(image, [320, 350], save=True)
