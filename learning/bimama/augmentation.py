import cv2 as cv
import numpy as np
import torch


class Sequential:
    def __init__(self, augmentors, random_order=False):
        self.augmentors = augmentors
        self.random_order = random_order

    def __call__(self, image, poses):
        for a in self.augmentors:
            image, poses = a(image, poses)
        return image, poses


class Affine:
    def __init__(self, x_percent=None, y_percent=None, rotate=None, scale_percent=None):
        self.x_percent = x_percent
        self.y_percent = y_percent
        self.rotate = rotate
        self.scale_percent = scale_percent

    @staticmethod
    def get_rotation_matrix(x, y, theta, center, scale):
        t = cv.getRotationMatrix2D(center, -theta * 180.0 / np.pi, scale)
        t[0, 2] += x
        t[1, 2] += y
        return t

    @staticmethod
    def rand_uniform(low=0.0, high=1.0):
        return low + torch.rand(1)[0].numpy() * (high - low)

    @staticmethod
    def is_pose_inside(pose, image_size, offset=8):
        return (offset <= pose[0] < image_size[1] - offset) and (offset <= pose[1] < image_size[0] - offset)

    def __call__(self, image, poses, max_tries=3):
        center = [image.shape[1] / 2, image.shape[0] / 2]

        color = (image[0, 0] + image[-1, -1] + image[0, -1] + image[-1, 0]) / 4
        color = (float(color[0]), float(color[1]))

        found_trans = False
        for _ in range(max_tries):
            theta_trans = self.rand_uniform(low=self.rotate[0], high=self.rotate[1]) if self.rotate else 0.0
            x_trans = self.rand_uniform(low=self.x_percent[0] * image.shape[1], high=self.x_percent[1] * image.shape[1]) if self.x_percent else 0.0
            y_trans = self.rand_uniform(low=self.y_percent[0] * image.shape[0], high=self.y_percent[1] * image.shape[0]) if self.y_percent else 0.0
            scale_trans = self.rand_uniform(low=self.scale_percent[0], high=self.scale_percent[1]) if self.scale_percent else 1.0

            random_trans = self.get_rotation_matrix(x_trans, y_trans, theta_trans, center, scale_trans)
            poses_trans = np.copy(poses)
            for p in poses_trans:
                p[:2] = random_trans @ [p[0], p[1], 1]
                p[2] = p[2] + theta_trans

            if len(poses) == 0 or (self.is_pose_inside(poses_trans[0], image.shape) and self.is_pose_inside(poses_trans[1], image.shape)):
                poses = poses_trans
                found_trans = True
                break

        if found_trans:
            image = cv.warpAffine(image, random_trans, (image.shape[1], image.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=color)
        return image, poses


class Flip:
    def __init__(self, lr_percent=None, ud_percent=None):
        self.lr_percent = lr_percent
        self.ud_percent = ud_percent

    def __call__(self, image, poses):
        p_ud, p_lr = torch.rand(2)

        if p_ud < self.ud_percent:
            image = cv.flip(image, 0) # UD
            for p in poses:
                p[1] = image.shape[0] - p[1]
                p[2] = np.pi - p[2]
        
        if p_lr < self.lr_percent:
            image = cv.flip(image, 1) # LR
            for p in poses:
                p[0] = image.shape[1] - p[0]
                p[2] = -p[2]

        return image, poses


class Color:
    def __init__(self, brightness_change=None, contrast_change=None):
        self.brightness_change = brightness_change
        self.contrast_change = contrast_change

    @staticmethod
    def rand_uniform(low=0.0, high=1.0):
        return low + torch.rand(1)[0].numpy() * (high - low)

    def __call__(self, image, poses):
        alpha = self.rand_uniform(1.0 - self.contrast_change, 1.0 + self.contrast_change) if self.contrast_change else 1.0
        beta = self.rand_uniform(-self.brightness_change * 255, self.brightness_change * 255) if self.brightness_change else 0.0
        image[:, :, 0] = np.clip(alpha * image[:, :, 0] + beta, 0, 255).astype(np.uint8)
        return image, poses
