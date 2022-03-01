import cv2 as cv
import numpy as np
import pyclipper


def find_first_point_in_contour(start, end, contour, resolution=1.0):
    start = start.astype(np.float32)
    end = end.astype(np.float32)
    diff_normalized = (end - start) / np.linalg.norm(end - start, 2)

    x = start
    dis = cv.pointPolygonTest(contour, x, True)
    while abs(dis) > resolution:
        x -= dis * diff_normalized
        dis = cv.pointPolygonTest(contour, x, True)

    return x


def get_normal_theta_at_contour_index(i, contour):
    center_point = contour[i][0]
    weights = [1, 2, 3, 3, 2, 1]
    angles = [np.arctan2(*(np.sign(diff_index)*(contour[(i + diff_index) % len(contour)][0] - center_point))) for diff_index in [-3, -2, -1, 1, 2, 3]]
    return np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights)) - np.pi/2 # Weightd circular mean


def maximize_area_of_polygon_along_contour(contour, point_left, point_right, mask_path):
    mask_path = mask_path if isinstance(mask_path, tuple) else (mask_path,)
    
    hull = contour.squeeze(1)
    for m in mask_path:
        pc = pyclipper.Pyclipper()
        pc.AddPath(hull, pyclipper.PT_SUBJECT)
        pc.AddPath(m, pyclipper.PT_CLIP)
        hull = pc.Execute(pyclipper.CT_INTERSECTION)[0]

    hull = np.expand_dims(np.array(hull), 1)

    max_points_index = None
    max_points_area = 0.0
    for i, h1 in enumerate(hull):
        for j in range(i+1, len(hull)):
            h2 = hull[j]

            area = max(
                abs(pyclipper.Area([point_left, h1[0], h2[0], point_right])),
                abs(pyclipper.Area([point_right, h1[0], h2[0], point_left])),
            )
            if area > max_points_area:
                h1_idx = np.argmin(np.linalg.norm(contour - h1, axis=2))
                h2_idx = np.argmin(np.linalg.norm(contour - h2, axis=2))

                max_points_index = h1_idx, h2_idx
                max_points_area = area

    return max_points_index
