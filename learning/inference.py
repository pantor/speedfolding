from time import time
from typing import Dict

from autolab_core import RgbdImage, RigidTransform
import cv2 as cv
from loguru import logger
import numpy as np
import torch
from torchvision import transforms

from classification.model import PrimitivesClassifier, FlingToFoldClassifier
from database import Database
from drawing import draw_pose_circle, draw_mask, draw_filled_mask
from heuristics.planar_transform import PlanarTransform
from reward import segment

from bimama.model import BiMamaNet
from bimama.prediction import Prediction


class Inference:
    def __init__(self, multi_model_name, primitives_model_name=None, experiment=None, is_depth_image_scaled=False):
        self.experiment = experiment

        self.db = Database('test')
        self.model_base_path = self.db.base_path / 'models'
        self.multi_model_name = multi_model_name
        self.primitives_model_name = primitives_model_name

        self.img_size = (256, 192)

        self.left_mask_path = np.array([(0, 174), (0, 36), (24, 24), (48, 18), (96, 24), (148, 48), (186, 75), (231, 148), (236, 186)])
        self.right_mask_path = np.array([(30, 176), (35, 150), (70, 77), (96, 52), (160, 26), (255, 22), (255, 186)])

        self.label_to_primitive = ['fling', 'fling-to-fold'] if 'f2f' in primitives_model_name else ['fling', 'pick-and-hold', 'drag', 'done']
        self.is_depth_image_scaled = is_depth_image_scaled

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)

        multi = BiMamaNet(num_rotations=20)
        multi.load_state_dict(torch.load(self.model_base_path / multi_model_name))
        self.prediction = Prediction(multi)

        if primitives_model_name:
            self.primitives = FlingToFoldClassifier() if 'f2f' in primitives_model_name else PrimitivesClassifier()
            self.primitives.load_state_dict(torch.load(self.model_base_path / primitives_model_name))
            self.primitives.eval()
            if use_cuda:
                self.primitives = self.primitives.cuda()

        self.transform = transforms.ToTensor()

    def transform_input(self, image: RgbdImage):
        color_image = image.raw_data[:, :, 0] / 255

        if self.is_depth_image_scaled:
            depth_image = image.raw_data[:, :, 3] / 255
        
        else:
            depth_image = image.raw_data[:, :, 3]

            min_distance, max_distance = 0.85, 1.25

            depth_image[depth_image == 0.0] = max_distance
            depth_image = np.clip((max_distance - depth_image) / (max_distance - min_distance), 0.0, 1.0)

        img = np.stack([color_image, depth_image], axis=-1).astype(np.float32)
        img = img[:-116, 145:-12]
        img = cv.resize(img, self.img_size)
        return img

    @classmethod
    def pose_to_planar_transform(cls, pose, image_shape):  # Scale poses to relative size
        x = (pose[0] / image_shape[1] * (1032 - 145 - 12) + 145) / 1032
        y = pose[1] / image_shape[0] * (772 - 116) / 772
        return PlanarTransform(position=[float(x), float(y)], theta=float(pose[2]))

    def is_action_executable(self, action_type: str, pixel1, pixel2, image, points_3d, verbose=True) -> Dict[str, RigidTransform]:
        """calculates the 3D RigidTransforms points, returns None if action is not executable"""

        pick1 = self.experiment.pixel_to_transform(pixel1, image.shape, points_3d)
        pick2 = self.experiment.pixel_to_transform(pixel2, image.shape, points_3d)
        
        if not self.experiment.is_pose_within_workspace(pick1) or not self.experiment.is_pose_within_workspace(pick2):
            if verbose:
                logger.error(f"Skip pose as it's outside workspace: {pick1.translation}, {pick2.translation}")
            return None

        if action_type == 'fling' or action_type == 'fling-to-fold' or action_type == 'drag':
            pick_left, pick_right = self.experiment.assign_to_arm(pick1, pick2)
            pick_left, pick_right, left_sign, right_sign, left_place_diff, right_place_diff = self.experiment.optimize_angle_a_for_reachability(pick_left, pick_right)

            if pick_left.translation[1] < -0.24 or pick_right.translation[1] > 0.24:
                if verbose:
                    logger.warning(f"Left or right pick point is too far to the other side: {pick_left.translation} | {pick_right.translation}")
                return None

            # Check not only for pick points, but also for rotated and shifted grasp pose
            joints_left, joints_right = self.experiment.model.ik(pick_left, pick_right)
            if joints_left is None or joints_right is None:
                if verbose:
                    logger.warning(f"Could not find inverse kinematic to fling poses: {pick_left.translation} | {pick_right.translation}")
                return None

            test_left = pick_left * self.experiment.get_transform(b=left_sign*0.3, frame='l_tcp')
            test_right = pick_right * self.experiment.get_transform(b=right_sign*0.3, frame='r_tcp')
            joints_left2, joints_right2 = self.experiment.model.ik(test_left, test_right)

            if joints_left2 is None or joints_right2 is None:
                if verbose:
                    logger.warning(f"Could not find inverse kinematic to fling poses: {pick_left.translation} | {pick_right.translation}")
                return None

            return {'pick_left': pick_left, 'pick_right': pick_right, 'left_sign': left_sign, 'right_sign': right_sign, 'left_place_diff': left_place_diff, 'right_place_diff': right_place_diff, 'pick1': pick1, 'pick2': pick2}

        elif action_type == 'pick-and-hold':
            place_pixel, hold_pixel = self.experiment.hph.refine_place_and_hold_points(pixel1, pixel2, image=image.color.raw_data)

            pick = pick1
            place = self.experiment.pixel_to_transform(place_pixel, image.shape, points_3d)
            hold = self.experiment.pixel_to_transform(hold_pixel, image.shape, points_3d) if hold_pixel is not None else None

            data_left, data_right = self.experiment.assign_to_arm((pick, place), hold)
            pick_with_left_arm = isinstance(data_left, tuple)

            pick_sign = 1.0
            hold_sign = -1 if pick_with_left_arm else 1
            
            transforms = {'pick': pick, 'place': place, 'hold': hold}
            if pick_with_left_arm:
                (pick, place), hold = data_left, data_right
                if hold:
                    pick, hold, pick_sign, _, _, _ = self.experiment.optimize_angle_a_for_reachability(pick, hold)
                if pick_sign < 0.0:
                    place = place * self.experiment.get_transform(c=np.pi, frame=place.from_frame)

                transforms['pick_left'] = pick
                transforms['place_left'] = place
                transforms['hold_right'] = hold
                transforms['pick_sign'] = pick_sign

                joints_left, joints_right = self.experiment.model.ik(pick, hold)
                hold_side = hold * self.experiment.get_transform(a=hold_sign * 0.5, frame='r_tcp') if hold else None
                joints_left2, joints_right2 = self.experiment.model.ik(place, hold_side)

                if joints_left is None or joints_left2 is None:
                    if verbose:
                        logger.warning(f"Could not find inverse kinematic to pick-and-hold poses: {pick.translation} | {place.translation} | {hold.translation}")
                    return None
                if hold is not None and (joints_right is None or joints_right2 is None):
                    logger.warning(f"Hold pose is out of reach (so continue without hold): {hold.translation}")
                    transforms['hold_right'] = None
            else:
                hold, (pick, place) = data_left, data_right
                if hold:
                    hold, pick, _, pick_sign, _, _ = self.experiment.optimize_angle_a_for_reachability(hold, pick)
                if pick_sign < 0.0:
                    place = place * self.experiment.get_transform(c=np.pi, frame=place.from_frame)

                transforms['pick_right'] = pick
                transforms['place_right'] = place
                transforms['hold_left'] = hold
                transforms['pick_sign'] = pick_sign

                joints_left, joints_right = self.experiment.model.ik(hold, pick)
                hold_side = hold * self.experiment.get_transform(a=hold_sign * 0.5, frame='l_tcp') if hold else None
                joints_left2, joints_right2 = self.experiment.model.ik(hold_side, place)

                if hold is not None and (joints_left is None or joints_left2 is None):
                    logger.warning(f"Hold pose is out of reach (so continue without hold): {hold.translation}")
                    transforms['hold_left'] = None
                if joints_right is None or joints_right2 is None:
                    if verbose:
                        logger.warning(f"Could not find inverse kinematic to pick-and-hold poses: {pick.translation} | {place.translation} | {hold.translation}")
                    return None

            return transforms

        raise Exception(f'[inference checker] action type {action_type} not implemented yet.')

    def should_move_for_action(self, info, verbose=True) -> bool:
        return info['rect'][1] + info['rect'][3] < 550 and info['y'] < 0.42

    def predict_action(self, image: RgbdImage, selection=None, action_type: str = None, save=False):
        timing = {}
        start = time()

        img = self.transform_input(image)
        _, info = segment(image.color.raw_data)

        left_mask = draw_filled_mask(self.left_mask_path, img.shape[:2])
        right_mask = draw_filled_mask(self.right_mask_path, img.shape[:2])
        # cv.imwrite('data/current/test.png', 255*left_mask)

        pre_processing = time()
        timing['pre_processing'] = pre_processing - start
        
        # Predict primitive action type
        confidences = self.predict_primitive_confidences(image)
        if action_type == None:
            action_type = self.label_to_primitive[int(np.argmax(confidences))]

        if self.experiment:
            points_3d = self.experiment.get_pointcloud(image)
        
        changed_drag = False
        if action_type == 'drag':
            action_type = 'fling-to-fold'
            changed_drag = True

        action_time = time()
        timing['action_type'] = action_time - pre_processing

        action_iterator, heatmaps, nn_timing = self.prediction.predict(img, selection=selection, action_type=action_type, left_mask=left_mask, right_mask=right_mask, return_timing=True)

        nn_prediction = time()
        timing['nn'] = nn_timing

        if changed_drag:
            action_type = 'drag'

        for i, (_, poses, scores) in enumerate(action_iterator()):
            if action_type == 'done' or not self.experiment:
                transforms = None
                break
            if action_type == 'fling-to-fold' and i > 1:
                return {'type': 'f2f-out-of-reach'}

            pixel1 = PlanarTransform.fromRelativePixelDictionary(self.pose_to_planar_transform(poses[0], img.shape).toDict(), image.shape)
            pixel2 = PlanarTransform.fromRelativePixelDictionary(self.pose_to_planar_transform(poses[1], img.shape).toDict(), image.shape)

            verbose = (i % 100 == 0)

            transforms = self.is_action_executable(action_type, pixel1, pixel2, image, points_3d, verbose=verbose)
            if transforms:
                break

            if i > 20 and self.should_move_for_action(info, verbose=verbose):
                action_type = 'move-to-center'
                break

            if verbose:
                logger.warning(f'Skip pose as it is not safe.')

        selection = time()
        timing['selection'] = selection - nn_prediction

        assert poses is not None, f'Could not find a valid pose (end of iterator).'

        if save:
            img_draw = np.copy(img)

            img_draw = (img_draw[:, :, 0] * 255).astype(np.uint8)
            img_draw = cv.cvtColor(img_draw, cv.COLOR_GRAY2BGR)

            heatmaps['pick-and-hold'] = heatmaps['pick'] + heatmaps['place']
            heatmaps['fling-to-fold'] = heatmaps['fling-to-fold-sleeve'] + heatmaps['fling-to-fold-bottom']
            heatmaps['drag'] = heatmaps['fling-to-fold']
            heat = heatmaps[action_type] if action_type in heatmaps else np.zeros_like(heatmaps['fling'])

            vis = np.mean(heat, axis=0)
            vis = cv.normalize(vis, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            vis = cv.applyColorMap(vis, cv.COLORMAP_JET)

            overlay = cv.addWeighted(img_draw, 0.65, vis, 0.35, 0)

            draw_mask(overlay, mask=self.left_mask_path, color=(100, 220, 255), polygon=True, fill=True)
            draw_mask(overlay, mask=self.right_mask_path, color=(255, 220, 100), polygon=True, fill=True)
            draw_pose_circle(overlay, poses[0], (255, 255, 255))
            draw_pose_circle(overlay, poses[1], (255, 255, 255), rect=True)
            cv.imwrite('data/current/heatmap.png', overlay)

        score = np.mean(scores)
        poses = [self.pose_to_planar_transform(p, img.shape).toDict() for p in poses]

        total = time()
        timing['total'] = total - start
        
        return {'type': action_type, 'poses': poses, 'transforms': transforms, 'score': score, 'is_self_supervised': True, 'is_human_annotated': False, 'needs_annotation': False, 'timing': timing}

    def predict_primitive_confidences(self, image: RgbdImage):
        assert self.primitives_model_name, "No primitive model was loaded"

        img = self.transform_input(image)
        img = np.stack([img[:, :, 0], img[:, :, 0], img[:, :, 1]], axis=-1).astype(np.float32)
        img_t = self.transform(img).cuda() 
        imgs = img_t.view(-1, img_t.shape[0], img_t.shape[1], img_t.shape[2])

        result = self.primitives.forward(imgs)
        return result.detach().cpu().numpy()[0]

    def predict_primitive(self, image: RgbdImage) -> str:
        confidences = self.predict_primitive_confidences(image)
        res = np.argmax(confidences)
        return self.label_to_primitive[int(res)]
