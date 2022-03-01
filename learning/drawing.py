from typing import Dict

import cv2 as cv
import numpy as np


def get_rotation_matrix(theta: float):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def draw_pose(image, pose: Dict, label=None):
    color = (20, 20, 255)
    center = np.array([pose['x'] * image.shape[1], pose['y'] * image.shape[0]])
    handle = center + get_rotation_matrix(pose['theta']) @ np.array([20, 0])

    left = center + get_rotation_matrix(pose['theta']) @ np.array([0, 40])
    right = center + get_rotation_matrix(pose['theta']) @ np.array([0, -40])
    left_up = center + get_rotation_matrix(pose['theta']) @ np.array([8, 40])
    left_down = center + get_rotation_matrix(pose['theta']) @ np.array([-8, 40])
    right_up = center + get_rotation_matrix(pose['theta']) @ np.array([8, -40])
    right_down = center + get_rotation_matrix(pose['theta']) @ np.array([-8, -40])

    cv.line(image, center.astype(int), handle.astype(int), color, 2)
    cv.line(image, left.astype(int), right.astype(int), color, 3)
    cv.line(image, left_up.astype(int), left_down.astype(int), color, 3)
    cv.line(image, right_up.astype(int), right_down.astype(int), color, 3)

    if label:
        image = cv.putText(image, label, (center + [-4, -15]).astype(int), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv.LINE_AA)


def draw_pose_circle(img, pose, color, rect=False):
    center = pose[:2]
    handle = center + get_rotation_matrix(pose[2]) @ np.array([8, 0])

    if rect:
        cv.rectangle(img, (center - [2, 2]).astype(int), (center + [2, 2]).astype(int), color, 1)
    else:
        cv.circle(img, center.astype(int), 3, color, 1)

    cv.line(img, center.astype(int), handle.astype(int), color, 1)


def draw_action(image, action):
    if 'poses' not in action:
        return

    ps = action['poses']
    for p, l in zip(ps, ['Pick 1', 'Pick 2']):
        draw_pose(image, p, label=l)


def draw_masked_area(image, mask_nn_input, mask_heatmap, color=(0, 255, 255)):
    # Input NN
    input_nn_tl = np.array([mask_nn_input['left'], mask_nn_input['top']])
    input_nn_br = image.shape[1::-1] - np.array([mask_nn_input['right'], mask_nn_input['bottom']])
    cv.rectangle(image, input_nn_tl.astype(int), input_nn_br.astype(int), color, 1)

    # Masked Heatmap
    rescaled_size = np.array([192, 256])
    new_size = image.shape[:2] - np.array([mask_nn_input['top'] + mask_nn_input['bottom'], mask_nn_input['left'] + mask_nn_input['right']])
    factor = new_size / rescaled_size
    mask_tl = input_nn_tl + np.array([mask_heatmap['left'], mask_heatmap['top']]) * factor
    mask_br = input_nn_br - np.array([mask_heatmap['right'], mask_heatmap['bottom']]) * factor
    cv.rectangle(image, mask_tl.astype(int), mask_br.astype(int), color, 1)


def draw_mask(image, mask, color=(0, 255, 255), polygon=False, fill=False):
    if polygon:
        cv.polylines(image, [mask], True, color, thickness=1)

        if fill:
            draw_contour = np.copy(image)
            cv.drawContours(draw_contour, [mask], 0, color, -1)
            cv.addWeighted(image, 0.8, draw_contour, 0.2, 0, dst=image)
        return

    mask_tl = np.array([mask['left'], mask['top']])
    mask_br = image.shape[1::-1] - np.array([mask['right'], mask['bottom']])
    cv.rectangle(image, mask_tl.astype(int), mask_br.astype(int), color, 1)


def draw_filled_mask(mask_path, shape, blur=None):
    image = np.zeros(shape=shape, dtype=np.float32)
    cv.drawContours(image, [mask_path], 0, (1.0), -1)
    if blur is not None:
        image = cv.GaussianBlur(image, (blur, blur), 0)
    return image
