from datetime import datetime
import json

import cv2 as cv
import numpy as np
import requests

from autolab_core import RgbdImage


class Episode:
    def __init__(self):
        self.id = self.generate_id()
        self.save = True
        self.actions = []

    @classmethod
    def generate_id(cls):
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]


class Saver:
    """This uploads actions and images to the database."""
    def __init__(self, url: str):
        self.url = url

    @classmethod
    def _jsonify_episode(cls, x):
        # Remove transforms as RigidTransforms are not serializable (yet) and we save every pose in 2D so far
        for a in x.actions:
            a.pop('transforms', None)
        return x.__dict__

    def save_action(self, collection: str, episode_id: str, action_id: int, data):
        try:
            r = requests.post(self.url + 'action', json={
                'collection': collection,
                'episode_id': episode_id,
                'action_id': action_id,
                'data': json.dumps(data, default=Saver._jsonify_episode),
            })
            r.raise_for_status()
        except requests.exceptions.RequestException:
            raise Exception('Could not save action result!')

    def save_split_image(self, image, collection: str, episode_id: str, action_id: int, scene: str, camera: str):
        """saves an image given by a numpy array / cv mat"""

        image_data = cv.imencode('.png', image)[1].tobytes()

        try:
            r = requests.post(self.url + 'image', files={
                'file': ('image.png', image_data, 'image/png', {'Expires': '0'})
            }, data={
                'collection': collection,
                'episode_id': episode_id,
                'action_id': action_id,
                'scene': scene,
                'camera': camera,
            })
            r.raise_for_status()
        except requests.exceptions.RequestException:
            raise Exception('Could not save image!')

    def save_image(self, image: RgbdImage, collection: str, episode_id: str, action_id: int, scene: str, camera: str = None):
        color_camera = f'{camera}-color' if camera else 'color'
        depth_camera = f'{camera}-depth' if camera else 'depth'

        color_image = image.raw_data[:, :, :3]
        depth_image = image.raw_data[:, :, 3]

        min_distance, max_distance = 0.85, 1.25

        depth_image[depth_image == 0.0] = max_distance
        depth_image = np.clip((max_distance - depth_image) / (max_distance - min_distance), 0.0, 1.0) * 255

        self.save_split_image(color_image, collection, episode_id, action_id, scene, color_camera)
        self.save_split_image(depth_image, collection, episode_id, action_id, scene, depth_camera)
