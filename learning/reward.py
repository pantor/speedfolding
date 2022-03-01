import cv2 as cv
import numpy as np


def segment(color_image, return_contour=False, remove_bottom=True, thresh_value=90):
    if len(color_image.shape) == 3:
        img_gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    else:
        img_gray = color_image
    
    _, img_thresh = cv.threshold(img_gray, thresh_value, 255, 0)
    
    img_thresh = cv.erode(img_thresh, np.ones((8, 8), np.uint8))
    img_thresh = cv.dilate(img_thresh, np.ones((16, 16), np.uint8))
    img_thresh = cv.erode(img_thresh, np.ones((8, 8), np.uint8))

    # Remove robot at the bottom of image
    if remove_bottom:
        img_thresh[-100:, :] = 0

    contours, _ = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_index = np.argmax([cv.contourArea(c) for c in contours])
    all_contour = contours[contour_index]

    result = np.zeros_like(img_thresh)
    cv.fillPoly(result, [all_contour], 255)

    moments = cv.moments(all_contour)
    area = moments['m00'] / (color_image.shape[0] * color_image.shape[1])
    x = (moments['m10'] / moments['m00']) / color_image.shape[1]
    y = (moments['m01'] / moments['m00']) / color_image.shape[0]
    bounding_rect = cv.boundingRect(all_contour)

    info = {'area': area, 'x': x, 'y': y, 'rect': bounding_rect}

    if return_contour:
        return result, info, all_contour
    return result, info


def calculate_reward(mask_before, mask_after):
    area_before = np.mean(mask_before) / 255
    area_after = np.mean(mask_after) / 255

    # with a magic number to get meaningful range between -1.0 and 1.0
    # reward = np.tanh((area_after - area_before) * 14)
    reward = np.tanh(area_before * (area_after - area_before) * 64)
    return reward


def calculate_reward_change_converage_and_confidence(inference, image_before, image_after):
    mask_before, _ = segment(image_before.color.raw_data)
    mask_after, _ = segment(image_after.color.raw_data)
    
    area_before = np.mean(mask_before) / 255
    area_after = np.mean(mask_after) / 255

    before_done_confidence = inference.predict_primitive_confidences(image_before)
    after_done_confidence = inference.predict_primitive_confidences(image_after)
    
    confidence_change = after_done_confidence[2] - before_done_confidence[2]
    coverage_change = area_after - area_before

    return np.tanh(2 * coverage_change + 1 * confidence_change)
