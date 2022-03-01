from pathlib import Path
import sys
from typing import List

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.absolute().parent))

import augmentation as iaa

sys.path.insert(0, str(Path(__file__).parent.absolute().parent.parent))

from database import Database


class BiFoldingDataset(Dataset):
    def __init__(self, collection: str, actions: List[any], use_augmentation=False):
        self.db = Database(collection, base_path=Path('tmp'))
        self.resized_image_path = self.db.base_path / f'{collection}-resized'
        self.actions = actions

        self.num_rotations = 20
        self.image_size = (256, 192)
        self.sigma = 3
        self.sigma_orientation = 0.1 * self.sigma

        self.transform_action = None
        if use_augmentation:
            self.transform_action = iaa.Sequential([
                iaa.Flip(lr_percent=0.5, ud_percent=0.25),
                iaa.Affine(
                    x_percent=(-0.25, 0.25),
                    y_percent=(-0.25, 0.25),
                    rotate=(-np.pi/2, np.pi/2),
                    scale_percent=(0.9, 1.1),
                ),
                iaa.Color(brightness_change=0.2, contrast_change=0.25),
            ], random_order=False)

        self.transform_to_tensor = transforms.ToTensor()

        self.primitive_classes = ['fling', 'fling-to-fold', 'pick-and-hold', 'drag', 'done']
        total_counts = {p: len(list(filter(lambda a: a['type'] == p, self.actions))) for p in self.primitive_classes}
        total_annotations_count = {p: sum(map(lambda a: float(a['is_human_annotated']), filter(lambda a: a['type'] == p, self.actions))) for p in self.primitive_classes}
        total_rewards = {p: sum(map(lambda a: 0.8 if a['is_human_annotated'] else max(a['reward'], 0.0) if 'reward' in a else 0.0, filter(lambda a: a['type'] == p, self.actions))) for p in self.primitive_classes}

        print(f'Init dataset with {len(self.actions)} actions')
        print(f"Count ratios for " + ', '.join([f'{k}: {v/len(self.actions):0.3f}' for k, v in total_counts.items()]))
        # print(f"Count ratios for fling: {total_counts['fling']/len(self.actions):0.3f}, pick-and-hold: {total_counts['pick-and-hold']/len(self.actions):0.3f}, done: {total_counts['done']/len(self.actions):0.3f}")
        print(f"Total reward for fling: {total_rewards['fling']:0.3f}, pick-and-hold: {total_rewards['pick-and-hold']:0.3f}, done: {total_rewards['done']:0.3f}")
        print(f"Balance factor all fling: {1.0:0.3f}, pick-and-hold: {total_counts['fling']/total_counts['pick-and-hold']:0.3f}, done: {total_counts['fling']/total_counts['done']:0.3f}")
        print(f"Balance factor human annotation fling: {1.0:0.3f}, pick-and-hold: {total_annotations_count['fling']/total_annotations_count['pick-and-hold']:0.3f}, done: {total_annotations_count['fling']/total_annotations_count['done']:0.3f}")

        self.resize_images_for_actions(update_only=True)

    @staticmethod
    def load_train_test_actions(collection: str, train_test_split=0.8):
        db = Database(collection, base_path=Path('tmp'))
        train_actions, test_actions = [], []

        where_clause = db.get_where_clause(needs_annotation=False)
        
        for action in db.yield_actions(where_clause=where_clause):
            l = train_actions if db.binary_decision(action['episode_id'], train_test_split) else test_actions
            l.append(action)

        return train_actions, test_actions

    def resize_images_for_actions(self, update_only=False):
        self.resized_image_path.mkdir(exist_ok=True)

        for a in self.actions:
            episode_path = self.resized_image_path / a['episode_id']
            if episode_path.exists() and update_only:
                continue
            episode_path.mkdir(exist_ok=True)

            for camera in ['color', 'depth']:
                image = self.db.get_image(a['episode_id'], a['action_id'], scene='before', camera=camera)
                image = image[:-116,145:-12]
                image = cv.resize(image, self.image_size)
                cv.imwrite(str(episode_path / self.db.get_image_filename(a['action_id'], 'before', camera)), image)

    def read_image(self, episode_id: str, action_id: int):
        image = cv.imread(str(self.resized_image_path / episode_id / self.db.get_image_filename(action_id, 'before', 'color')), cv.IMREAD_GRAYSCALE)
        image_depth = cv.imread(str(self.resized_image_path / episode_id / self.db.get_image_filename(action_id, 'before', 'depth')), cv.IMREAD_GRAYSCALE)
        image = np.stack([image, image_depth], axis=-1)
        return image

    def gauss_2d_batch(self, k):
        TH, X, Y = torch.meshgrid([torch.arange(0, self.num_rotations), torch.arange(0, self.image_size[0]), torch.arange(0, self.image_size[1])])
        TH, X, Y = TH.transpose(1, 2), X.transpose(1, 2), Y.transpose(1, 2)
        
        return torch.exp(-((X - k[0])**2 + (Y - k[1])**2)/(2 * self.sigma**2) - (TH * 2 * np.pi / self.num_rotations - (k[2] % (2 * np.pi)))**2/(2 * self.sigma_orientation**2))

    def pose_to_array(self, pose):
        x = (pose['x'] * 1032 - 145) / (1032 - 145 - 12)
        y = (pose['y'] * 772) / (772 - 116)
        return [x * self.image_size[0], y * self.image_size[1], pose['theta']]
        # return [pose['x'] * self.image_size[0], pose['y'] * self.image_size[1], pose['theta']]

    def is_inside(self, pose):
        return (0 <= pose[0] < self.image_size[0]) and (0 <= pose[1] < self.image_size[1])

    def __getitem__(self, index):
        action = self.actions[index]
        image = self.read_image(action['episode_id'], action['action_id'])

        poses = np.array([self.pose_to_array(p) for p in action['poses']])        

        if self.transform_action:
            image, poses = self.transform_action(image=image, poses=poses)

            # Permute pick1 and pick2 for fling action
            if action['type'] == 'fling':
                if torch.rand(1) < 0.5:
                    poses = np.copy(poses[::-1])
        
        image = self.transform_to_tensor(image)
        poses = torch.tensor(poses).float()

        # if action['type'] == 'fling-to-fold':
        #     action['type'] = 'fling'

        assert action['type'] in self.primitive_classes, "Unknown action type"
        if 'is_self_supervised' in action and action['is_self_supervised']:
            assert 'reward' in action, f"Reward was not calculated for action {action['episode_id']} yet"

        zero = torch.zeros(20, 192, 256)
        fling1_heatmap = self.gauss_2d_batch(poses[0]) if action['type'] == 'fling' else zero
        fling2_heatmap = self.gauss_2d_batch(poses[1]) if action['type'] == 'fling' else zero
        fling_to_fold1_heatmap = self.gauss_2d_batch(poses[0]) if action['type'] == 'fling-to-fold' else zero
        fling_to_fold2_heatmap = self.gauss_2d_batch(poses[1]) if action['type'] == 'fling-to-fold' else zero
        pnh_pick_heatmap = self.gauss_2d_batch(poses[0]) if action['type'] == 'pick-and-hold' else zero
        pnh_place_heatmap = self.gauss_2d_batch(poses[1]) if action['type'] == 'pick-and-hold' else zero

        reward = 0.8 if action['is_human_annotated'] and action['type'] != 'done' and action['type'] != 'drag' else max(action['reward'], 0.0) if 'reward' in action else 0.0
        primitive_index = self.primitive_classes.index(action['type'])

        if action['type'] != 'done' and action['type'] != 'drag' and (not self.is_inside(poses[0]) or not self.is_inside(poses[1])):
            reward = 0.0

        # Weight: [min weight, max weight], proportional to heatmap, normalized to 1 over batch
        if action['type'] == 'fling':
            fling_weight = torch.tensor([0.5, 1.0]) if action['is_human_annotated'] else torch.tensor([4e-2, 1.0])
            fling_to_fold_weight = torch.tensor([1e-6, 1e-6]) if action['is_human_annotated'] else torch.tensor([1e-6, 1e-6])
            pnh_weight = torch.tensor([0.5, 0.5]) if action['is_human_annotated'] else torch.tensor([1e-6, 1e-6])
        elif action['type'] == 'fling-to-fold':
            fling_weight = torch.tensor([1e-6, 1e-6]) if action['is_human_annotated'] else torch.tensor([1e-6, 1e-6])
            fling_to_fold_weight = torch.tensor([0.5, 1.0]) if action['is_human_annotated'] else torch.tensor([4e-2, 1.0])
            pnh_weight = torch.tensor([0.5, 0.5]) if action['is_human_annotated'] else torch.tensor([1e-6, 1e-6])
        elif action['type'] == 'pick-and-hold':
            fling_weight = torch.tensor([0.5, 0.5]) if action['is_human_annotated'] else torch.tensor([1e-6, 1e-6])
            fling_to_fold_weight = torch.tensor([0.5, 1.0]) if action['is_human_annotated'] else torch.tensor([4e-2, 1.0])
            pnh_weight = torch.tensor([1.0, 1.0]) if action['is_human_annotated'] else torch.tensor([4e-2, 1.0])
        elif action['type'] == 'drag':
            fling_weight = torch.tensor([0.5, 0.5])
            fling_to_fold_weight = torch.tensor([0.5, 0.5])
            pnh_weight = torch.tensor([0.5, 0.5])
        else: # done
            fling_weight = torch.tensor([0.5, 0.5])
            fling_to_fold_weight = torch.tensor([0.5, 0.5])
            pnh_weight = torch.tensor([0.5, 0.5])
        
        weights = torch.stack([fling_weight, fling_to_fold_weight, pnh_weight])
        annotation_weight = 1.0 if action['is_human_annotated'] else 0.01

        return image, fling1_heatmap, fling2_heatmap, fling_to_fold1_heatmap, fling_to_fold2_heatmap, pnh_pick_heatmap, pnh_place_heatmap, reward, weights, primitive_index, annotation_weight
    
    def __len__(self):
        return len(self.actions)


if __name__ == '__main__':
    _, test_actions = BiFoldingDataset.load_train_test_actions('test')
    test_dataset = BiFoldingDataset('test', test_actions, use_augmentation=False)
    train_dataset = BiFoldingDataset('test', test_actions, use_augmentation=True)

    index = 2

    samples = [test_dataset[index]] + [train_dataset[index] for _ in range(5)]

    for i, (image, heatmap_fling1, heatmap_fling2, heatmap_pick, heatmap_place, reward, weights, primitive_index, annotation_weight) in enumerate(samples):
        img = (255 * np.transpose(image.cpu().numpy(), (1, 2, 0))).astype(np.uint8)
        img = cv.cvtColor(img[:, :, 0], cv.COLOR_GRAY2BGR)

        heat = [heatmap_fling1 + heatmap_fling2, heatmap_pick + heatmap_place, np.zeros_like(heatmap_fling1)][primitive_index]
        heat = heat.mean(dim=0).cpu().numpy()
        heat = cv.normalize(heat, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        vis = cv.applyColorMap(heat, cv.COLORMAP_JET)
        img = cv.addWeighted(img, 0.65, vis, 0.35, 0)

        cv.imwrite(f'preds/data{i:05d}.png', img)
