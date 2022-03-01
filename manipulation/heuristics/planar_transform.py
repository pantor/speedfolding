import cv2 as cv
import numpy as np


class PlanarTransform:
    def __init__(self, position=None, theta=0.0):
        self.position = position
        self.theta = theta

    @staticmethod
    def fromRelativePixelDictionary(data, image_shape):
        return PlanarTransform(
            position=np.array([data['x'] * image_shape[1], data['y'] * image_shape[0]]),
            theta=-data['theta'],
        )

    @staticmethod
    def get_rotation_matrix(t: float):
        return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])

    def mirror_along_line(self, line):
        diff = np.array([line.start[1] - line.end[1], line.end[0] - line.start[0]])
        theta_line = np.arctan2(*diff)
        D = 2 * (diff[0] * (self.position[0] - line.start[0]) + diff[1] * (self.position[1] - line.start[1])) / np.linalg.norm(diff)**2
        return PlanarTransform(
            position=np.array([
                self.position[0] - diff[0] * D,
                self.position[1] - diff[1] * D
            ]),
            theta=(2*theta_line - self.theta)
        )

    def draw(self, image, color=(250, 100, 200)):
        cv.line(image, self.position.astype(int), (self.position + self.get_rotation_matrix(self.theta) @ np.array([24, 0])).astype(int), color, 2)
        cv.circle(image, self.position.astype(int), 4, color, -1)

    def toDict(self):
        return {'x': self.position[0], 'y': self.position[1], 'theta': self.theta}
