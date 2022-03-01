from autolab_core.image import RgbdImage, ColorImage, DepthImage

import json
import hashlib
import sqlite3
from pathlib import Path
import os

import cv2 as cv
import numpy as np


class Database:
    def __init__(self, collection: str):
        self.base_path = Path.home() / 'data'
        self.base_path.parent.mkdir(exist_ok=True, parents=True)

        self.collection = collection
        self.database_path = self.base_path / f'{collection}.db'
        is_new_database = not self.database_path.exists()

        self.conn = sqlite3.connect(self.database_path)
        self.conn.row_factory = sqlite3.Row
        self.c = self.conn.cursor()
        if is_new_database:
            self.create()

        self.action_list = None

    def create(self):
        (self.base_path / self.collection).mkdir()
        self.c.execute("CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id CHAR(20), action_id INTEGER, data JSON)")
        self.c.execute("CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id CHAR(20), action_id INTEGER, scene CHAR(20), camera CHAR(20), hash CHAR(64), UNIQUE(episode_id, action_id, scene, camera))")
        self.conn.commit()

    def load_action_list(self):
        self.c.execute(f"SELECT episode_id, action_id FROM actions ORDER BY episode_id DESC, action_id ASC")
        self.action_list = list(map(lambda e: (e[0], e[1]), self.c.fetchall()))

    @classmethod
    def get_where_clause(cls, needs_annotation: bool = None, is_human_annotated: bool = None, is_self_supervised: bool = None, is_grasp_failure: bool = None, primitive_type: str = None, reward: str = None, with_where=True):
        where_clauses = []
        if needs_annotation is not None:
            where_clauses.append(f"json_extract(data, '$.needs_annotation') == {int(needs_annotation)}")
        if is_human_annotated is not None:
            where_clauses.append(f"json_extract(data, '$.is_human_annotated') == {int(is_human_annotated)}")
        if is_self_supervised is not None:
            where_clauses.append(f"json_extract(data, '$.is_self_supervised') == {int(is_self_supervised)}")
        if is_grasp_failure is not None:
            where_clauses.append(f"json_extract(data, '$.is_grasp_failure') == {int(is_grasp_failure)}")
        if primitive_type is not None:
            where_clauses.append(f"json_extract(data, '$.type') == '{primitive_type}'")
        if reward is not None:
            if reward[0] in ['<', '>', '=']:
                where_clauses.append(f"json_extract(data, '$.reward') {reward}")

        where = 'WHERE ' if with_where else ''
        return where + '(' + ' AND '.join(where_clauses) + ')' if where_clauses else ''

    def yield_actions(self, needs_annotation: bool = None, is_human_annotated: bool = None, is_self_supervised: bool = None, is_grasp_failure: bool = None, primitive_type: str = None, reward: str = None, limit: int = None, skip: int = None, where_clause: str = None):
        limit = limit if limit else 1e12
        skip = skip if skip else 0
        where_clause = where_clause if where_clause else self.get_where_clause(needs_annotation, is_human_annotated, is_self_supervised, is_grasp_failure, primitive_type, reward) 

        self.c.execute(f"SELECT id as 'row_id', episode_id, action_id, data FROM actions {where_clause} ORDER BY episode_id DESC, action_id ASC LIMIT ? OFFSET ?", (limit, skip))
        for row in self.c.fetchall():
            action = dict(row)
            action.update(json.loads(row['data']))
            del action['data']
            yield action

    def get_action(self, episode_id: str, action_id: int):
        self.c.execute("SELECT id as 'row_id', episode_id, action_id, data FROM actions WHERE episode_id = ? AND action_id = ?", (episode_id, action_id))
        action = dict(self.c.fetchone())
        action.update(json.loads(action['data']))
        del action['data']
        return action

    @staticmethod
    def get_image_filename(action_id: int, scene: str, camera: str) -> str:
        return f'{action_id}-{scene}-{camera}.png'

    def get_image_path(self, episode_id: str, action_id: int, scene: str, camera: str) -> Path:
        if episode_id == 'current':
            return self.base_path / 'current' / f'image-{camera}.png'
        return self.base_path / self.collection / episode_id / self.get_image_filename(action_id, scene, camera)

    def get_image(self, episode_id: str, action_id: int, scene: str, camera: str):
        return cv.imread(str(self.get_image_path(episode_id, action_id, scene, camera)), cv.IMREAD_GRAYSCALE)

    def get_rgbd_image(self, episode_id: str, action_id: int, scene: str) -> RgbdImage:
        image_color = ColorImage.from_array(cv.imread(str(self.get_image_path(episode_id, action_id, scene, 'color'))))
        image_depth = DepthImage.from_array(cv.imread(str(self.get_image_path(episode_id, action_id, scene, 'depth')), cv.IMREAD_GRAYSCALE))
        return RgbdImage.from_color_and_depth(image_color, image_depth)

    def __getitem__(self, i: int):
        if not self.action_list:
            self.load_action_list()

        episode_id, action_id = self.action_list[i]
        image = self.get_image(episode_id, action_id, scene='before', camera='color')
        action = self.get_action(episode_id, action_id)
        poses = list(map(lambda e: [e['x'], e['y'], e['theta']], action['poses']))
        data = action['type'], np.array(poses)
        return np.array(image), data

    @staticmethod
    def binary_decision(string: str, p: float) -> bool:
        return float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16 < p


if __name__ == '__main__':
    db = Database('test')

    # 1. Example to iterate over all actions
    # You can also get specific types e.g. by primitive_type='pick-and-place'
    for action in db.yield_actions(is_human_annotated=True):
        image = db.get_image(action['episode_id'], action['action_id'], scene='before', camera='color')
        # print(image.shape, action)

        # Action is a dictionary with following fields:
        # - type: string enum either 'fling', 'pick-and-place', 'pick-and-hold', 'move', or 'done'
        # - poses: list of 2D Poses (each is a dictionary with {x, y, theta})

        if action['type'] != 'done':
            pick1, pick2 = action['poses']

    # 2. Get (image, action) pair by index
    image, action = db[0]
    print(image.shape, action)
