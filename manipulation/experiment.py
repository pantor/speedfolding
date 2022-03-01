from argparse import ArgumentParser
from pathlib import Path
import shutil
import time
from typing import Dict

import cv2 as cv
import numpy as np
from loguru import logger

from abb_librws import SGSettings
from autolab_core import RigidTransform, RgbdImage, DepthImage
from autolab_core import transformations as tr
from phoxi import PhoXiSensor
from yumi import YuMi

from heuristics.planar_transform import PlanarTransform
from heuristics.folding import FoldingHeuristic, Line
from heuristics.matching import Instruction, TemplateMatching
from heuristics.move import MoveHeuristic
from heuristics.hold_point import HoldPointHeuristic
from heuristics.utils import find_first_point_in_contour
from saver import Episode, Saver
from omply import RobotModel
from reward import segment


class Experiment:
    def __init__(self, seed=None, save_on_exit=True, on_exit_callback=None):
        self.saver = Saver(url='http://127.0.0.1:8000/api/')
        self.local_data_path = Path.home() / 'data' # Absolute path to database / logging directory
        self.rng = np.random.default_rng(seed)
        self.save_on_exit = save_on_exit
        self.on_exit_callback = on_exit_callback

        self.fh = FoldingHeuristic()
        self.mh = MoveHeuristic()
        self.hph = HoldPointHeuristic()
        self.tm = TemplateMatching()

        self.TIMEOUT = 7 # [s]
        self.T_CAM_BASE = RigidTransform.load(str(Path(__file__).parent.absolute().parent / 'data' / 'calibrations' / 'phoxi_to_world_bww.tf')).as_frames(from_frame='phoxi', to_frame='base_link')
        self.ABB_WHITE = RigidTransform(translation=[0, 0, 0.1325]) # [m]
        
        self.frame_left = self.ABB_WHITE.as_frames(RobotModel.l_tcp_frame, RobotModel.l_tip_frame)
        self.frame_right = self.ABB_WHITE.as_frames(RobotModel.r_tcp_frame, RobotModel.r_tip_frame)

        self.model = RobotModel()
        self.model.set_tcp(self.frame_left, self.frame_right)

        self.y = YuMi(l_tcp=self.frame_left, r_tcp=self.frame_right)

        # max_speed, hold_force, physical_limit
        self.y.left.gripper_settings = SGSettings(np.array([25, 20, 25], dtype=np.float64))
        self.y.left.gripper_settings = SGSettings(np.array([25, 20, 25], dtype=np.float64))

        self.full_speed = (0.6, 1.2 * np.pi)
        self.half_speed = (0.12, 0.24 * np.pi)
        self.fling_speed = (1.0, 1.0 * np.pi)
        self.drag_speed = (0.1, 0.2 * np.pi)
        self.stretch_speed = (0.06, 0.1 * np.pi)
        self.move_clothes_speed = (0.22, 0.3 * np.pi)

        self.in_grasp_distance = 0.018  # [m]

        self.cam = PhoXiSensor('1703005')
        self.cam.start()

        self.webcam = cv.VideoCapture(0)
        for _ in range(10):  # To adjust exposure
            self.webcam.read()

    def __del__(self):
        self.cam.stop()

    @classmethod
    def get_transform(cls, x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, c=0.0, to_frame='world', from_frame='unassigned', frame=None):
        to_frame = frame if frame else to_frame
        from_frame = frame if frame else from_frame
        return RigidTransform(translation=[x, y, z], rotation=tr.euler_matrix(a, b, c)[:3,:3], to_frame=to_frame, from_frame=from_frame)

    @classmethod
    def pixel_to_transform(cls, pixel_coordinate, image_shape, points_3d):
        # Or deal with PlanarTransform
        theta = 0.0
        if isinstance(pixel_coordinate, PlanarTransform):
            theta = pixel_coordinate.theta
            pixel_coordinate = pixel_coordinate.position

        idx = image_shape[1] * np.round(pixel_coordinate[1]) + np.round(pixel_coordinate[0])
        x, y, z = points_3d[int(idx)].vector
        return cls.get_transform(x, y, z, np.pi, 0.0, theta, from_frame='robot')

    @classmethod
    def assign_tcp_frame(cls, pose_left: RigidTransform, pose_right: RigidTransform):
        if isinstance(pose_left, tuple):
            for p in pose_left:
                p.from_frame = 'l_tcp'
        elif pose_left is not None:
            pose_left.from_frame = 'l_tcp'

        if isinstance(pose_right, tuple):
            for p in pose_right:
                p.from_frame = 'r_tcp'
        elif pose_right is not None:
            pose_right.from_frame = 'r_tcp'

    @classmethod
    def assign_to_arm(cls, pose1, pose2):
        """returns tuple with (left arm, right arm)"""

        pose_only1 = pose1[0] if isinstance(pose1, tuple) else pose1
        pose_only2 = pose2[0] if isinstance(pose2, tuple) else pose2

        trans1 = pose_only1.translation if pose_only1 is not None else np.zeros(3)
        trans2 = pose_only2.translation if pose_only2 is not None else np.zeros(3)

        max_to_1 = max(np.linalg.norm(trans1 - RobotModel.LEFT_BASE.translation), np.linalg.norm(trans2 - RobotModel.RIGHT_BASE.translation))
        max_to_2 = max(np.linalg.norm(trans2 - RobotModel.LEFT_BASE.translation), np.linalg.norm(trans1 - RobotModel.RIGHT_BASE.translation))
        
        if max_to_2 > max_to_1:
            cls.assign_tcp_frame(pose1, pose2)
            return pose1, pose2

        cls.assign_tcp_frame(pose2, pose1)
        return pose2, pose1

    @classmethod
    def is_pose_within_workspace(cls, pose: RigidTransform) -> bool:
        x = 0.12 < pose.translation[0] < 0.68
        y = -0.5 < pose.translation[1] < 0.5
        z = 0.035 < pose.translation[2] < 0.3
        return x and y and z

    @classmethod
    def optimize_angle_c(cls, pose: RigidTransform, frame: str):
        pose_sign = 1.0
        if abs(pose.euler_angles[2]) < np.pi/2:
            pose = pose * cls.get_transform(c=np.pi, frame=frame)
            pose_sign *= -1.0
        return pose, pose_sign

    @classmethod
    def optimize_angle_a_for_reachability(cls, left, right, relative_direction_matters=True):
        left, left_sign = cls.optimize_angle_c(left, frame='l_tcp')
        right, right_sign = cls.optimize_angle_c(right, frame='r_tcp')
        left_place_diff, right_place_diff = 0.0, 0.0

        distance = np.linalg.norm(left.translation - right.translation)
        center = left.interpolate_with(right, t=0.5)
        relative_left = left.inverse() * center
        relative_right = right.inverse() * center

        if relative_direction_matters:
            if relative_left.translation[1] > 0.03 and not -np.pi/2 < left.euler_angles[2] < np.pi/2 and left.translation[0] < 0.4:
                dc = -0.5 if left.translation[1] < -0.08 else -0.42 if left.translation[1] < -0.04 else -0.2 if left.translation[1] < 0.0 else 0.4
                left = left * cls.get_transform(c=-np.pi-dc, frame='l_tcp')
                left_sign *= -1.0

            # print(relative_right.translation, right.translation, right.euler_angles)
            if relative_right.translation[1] < -0.03 and not -np.pi/2 < right.euler_angles[2] < np.pi/2 and left.translation[0] < 0.4:
                dc = 0.5 if right.translation[1] > 0.08 else 0.42 if right.translation[1] > 0.04 else 0.2 if right.translation[1] > 0.0 else -0.4
                right = right * cls.get_transform(c=-np.pi-dc, frame='r_tcp')
                right_sign *= -1.0

            # Rotate gripper to increase reachability
            center = left.interpolate_with(right, t=0.5)
            relative_left = left.inverse() * center
            relative_right = right.inverse() * center

        # print(left.translation, right.translation, distance)
        
        if left.translation[0] > 0.54 and abs(left.inverse().translation[1]) > 0.1 and distance > 0.18: # Far away on x
            sgn = 1.0 if left.inverse().translation[1] < 0.0 else -1.0
            angle = 0.25 if left.translation[0] < 0.58 else 0.45
            left = left * cls.get_transform(a=-sgn*angle, frame='l_tcp')
        elif left.translation[0] > 0.6 and abs(left.inverse().translation[1]) > 0.1: # Far away on x
            sgn = 1.0 if left.inverse().translation[1] < 0.0 else -1.0
            left = left * cls.get_transform(a=-sgn*0.16, frame='l_tcp')
        elif left.translation[1] < -0.1: # Beneath the other arm
            pose_to_left = left.inverse() * RobotModel.LEFT_BASE
            sgn = 1.0 if pose_to_left.translation[1] > 0.0 else -1.0
            left = left * cls.get_transform(a=sgn*0.45, frame='l_tcp')
        elif left.translation[0] < 0.19 + min(left.translation[2], 0.1) and distance > 0.18: # Beneath its own shoulder
            sgn = 1.0 if left.inverse().translation[1] < -0.02 else -1.0
            left = left * cls.get_transform(a=sgn*0.35, frame='l_tcp')
        else:
            sgn = 0.0 if abs(relative_left.translation[1]) < 0.02 else 1.0 if relative_left.translation[1] < 0.0 else -1.0
            angle = 0.16 if distance > 0.1 else 0.24
            left = left * cls.get_transform(a=sgn*angle, frame='l_tcp')

        if right.translation[0] > 0.54 and abs(right.inverse().translation[1]) > 0.1 and distance > 0.18:
            sgn = 1.0 if right.inverse().translation[1] < 0.0 else -1.0
            angle = 0.25 if right.translation[0] < 0.58 else 0.45
            right = right * cls.get_transform(a=-sgn*angle, frame='r_tcp')
        elif right.translation[0] > 0.6 and abs(right.inverse().translation[1]) > 0.1:
            sgn = 1.0 if right.inverse().translation[1] < 0.0 else -1.0
            right = right * cls.get_transform(a=-sgn*0.16, frame='r_tcp')
        elif right.translation[1] > 0.1:
            pose_to_right = right.inverse() * RobotModel.RIGHT_BASE
            sgn = 1.0 if pose_to_right.translation[1] < 0.0 else -1.0
            right = right * cls.get_transform(a=-sgn*0.45, frame='r_tcp')
        elif right.translation[0] < 0.19 + min(right.translation[2], 0.1) and distance > 0.18:
            sgn = 1.0 if right.inverse().translation[1] < -0.02 else -1.0
            right = right * cls.get_transform(a=sgn*0.35, frame='r_tcp')
        else:
            sgn = 0.0 if abs(relative_right.translation[1]) < 0.02 else 1.0 if relative_right.translation[1] > 0.0 else -1.0
            angle = 0.16 if distance > 0.1 else 0.24
            right = right * cls.get_transform(a=-sgn*angle, frame='r_tcp')

        return left, right, left_sign, right_sign, left_place_diff, right_place_diff

    def take_image(self, save_as_current=True):
        """returns normal (projective) and orthographic image"""
        image_normal, image_ortho = self.cam.read_orthographic()
        self.cam.intrinsics = self.cam.create_intr(image_normal.width, image_normal.height)

        if save_as_current:
            self.saver.save_image(image_normal, 'test', 'current', 0, '.')

        return image_normal, image_ortho

    def take_image_for_annotation(self):
        episode = Episode()
        image_normal, image_ortho = self.take_image()

        self.saver.save_action('test', episode.id, len(episode.actions), data={'needs_annotation': True})
        self.saver.save_image(image_normal, 'test', episode.id, len(episode.actions), scene='before')
        self.saver.save_image(image_ortho, 'test', episode.id, len(episode.actions), scene='before', camera='ortho')

    def take_coverage(self):
        for _ in range(8):
            ret, colormap = self.webcam.read()
        
        if not ret:
            logger.error('camera not connected')

        mtx = np.array([[720.98017393, 0, 316.69994959], [0, 724.6546715, 226.36286998], [0, 0, 1]])
        dist = np.array([[-2.26828408e-01, -1.24263573e+00, 6.64668764e-03, 1.11834771e-04, 2.60003563e+00]])

        h,  w = colormap.shape[:2]
        newcameramtx, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        colormap = cv.undistort(colormap, mtx, dist, None, newcameramtx)

        _, info = segment(colormap, remove_bottom=False, thresh_value=130)
        return info['area']

    def home(self):
        self.check_for_wrist_joints()

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('home', 'check for wrist joints')

        self.y.left.open_gripper()
        self.y.right.open_gripper()

        self.y.move_joints_sync([RobotModel.L_HOME_STATE], [RobotModel.R_HOME_STATE], speed=self.full_speed, minimum_height=0.15)

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('home', 'joint motion')
    
    def hands_up(self):
        left_state = np.array([-1.46924693, -1.65620563, -0.96852145, 0.79790257, 1.18301064, -0.06245586, 0.27115401])
        right_state = np.array([1.32275243, -1.60578427, 0.95533947, 0.64276445, -1.44241998, -0.00299387, 0.03781762])
        self.check_for_wrist_joints()

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('home', 'check for wrist joints')

        self.y.left.open_gripper()
        self.y.right.open_gripper()

        self.y.move_joints_sync([left_state], [right_state], speed=self.full_speed, minimum_height=0.15)

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('home', 'joint motion')

    def exit_experiment(self, action_type='', message='', poses=None):
        """save information about the (undesired) exit due to a robot timeout"""
        if self.save_on_exit:
            exit_id = Episode.generate_id()
            logger.error(f'[exit {exit_id}] during {action_type}: {message}')

            # Save poses and relevant information
            path = self.local_data_path / 'exits' / exit_id
            path.mkdir()

            with open(path / 'information.txt', 'w') as info_file:
                info_file.write(f"{action_type}\n{message}\n")

            if poses:
                for i, p in enumerate(poses):
                    if not p: # Hold point might be None
                        continue
    
                    p.save(str(path / f'pose-{i}.tf'))

            # Copy images from current folder
            for f in ['heatmap.png', 'image-color.png', 'image-depth.png']:
                shutil.copy(self.local_data_path / 'current' / f, path / f)

        if self.on_exit_callback:
            self.on_exit_callback()
        
        exit()

    def get_pointcloud(self, image: RgbdImage):
        depth_copied = np.copy(image.depth.raw_data)
        depth_copied[depth_copied == 1.25] = 0.0
        depth_inpainted = DepthImage(depth_copied, frame=image.frame).inpaint(rescale_factor=0.25)
        return self.T_CAM_BASE * self.cam.intrinsics.deproject(depth_inpainted)

    def check_for_wrist_joints(self):
        left_joints = self.y.left.get_joints()
        if abs(left_joints[6]) > np.pi:
            left_joints[6] -= np.pi * np.sign(left_joints[6])
            self.y.left.move_joints_traj([left_joints], speed=self.full_speed)

        right_joints = self.y.right.get_joints()
        if abs(right_joints[6]) > np.pi:
            right_joints[6] -= np.pi * np.sign(right_joints[6])
            self.y.right.move_joints_traj([right_joints], speed=self.full_speed)

    def gen_random_scene(self, image: RgbdImage):
        mask, info = segment(image.color.data)
        # mask[:200, :] = 0.0
        H, W = mask.shape
        com = np.round((info['x'] * W, info['y'] * H)).astype(np.uint64)
        offset = 30
        rnd_x = self.rng.integers(-offset, offset)
        rnd_y = self.rng.integers(-offset, offset)
        pixel = com + np.array((rnd_x, rnd_y)).astype(np.uint64)
        # import cv2 as cv
        # img = cv.circle(mask, pixel, 3, (0,0,255), 3)
        # cv.imshow('img', img)
        # cv.waitKey()
        # import ipdb; ipdb.set_trace()
        # ys, xs = np.where(mask > 0)

        # offset = 1000
        # if len(ys) < 2*offset:
        #     return False
        
        # rnd = self.rng.integers(offset, len(ys)-offset)
        # pixel = (xs[rnd], ys[rnd])

        points_3d = self.get_pointcloud(image)
        pick = self.pixel_to_transform(pixel, image.shape, points_3d)

        if not self.is_pose_within_workspace(pick):
            logger.error('reset environment: pose not within workspace')
            return

        # move graps point down to increase probability of successful grasp
        pick = pick * self.get_transform(z=0.005, frame='robot')
        if pick.translation[2] > 0.075: # [m]
            pick = pick * self.get_transform(z=0.03, frame='robot')
        elif pick.translation[2] > 0.065: # [m]
            pick = pick * self.get_transform(z=0.02, frame='robot')
        elif pick.translation[2] > 0.055: # [m]
            pick = pick * self.get_transform(z=0.006, frame='robot')

        if abs(pick.euler_angles[2]) < np.pi/2:
            pick = pick * self.get_transform(c=np.pi, frame='robot')

        # assign arm
        y = pick.translation[1]
        arm = self.y.right if y < 0 else self.y.left
        
        # grasp the pick point
        pick_top = pick * self.get_transform(z=-0.04, frame='robot')
        arm.move_cartesian_traj([pick_top, pick], speed=self.full_speed, zone='z20')
        arm.sync()
        time.sleep(0.3)

        arm.close_gripper()
        arm.sync()
        time.sleep(0.4)

        # move shirt up and release
        sgn = -1.0 if y < 0 else 1.0
        release_pose = self.get_transform(
            x=self.rng.uniform(0.35, 0.45), y=self.rng.uniform(-0.1, 0.1), z=self.rng.uniform(0.4, 0.45),
            a=np.pi + self.rng.uniform(-0.1, 0.1) + sgn*0.5, b=self.rng.uniform(-0.1, 0.1), c=sgn * np.pi + self.rng.uniform(-0.2, 0.2),
            from_frame='robot'
        )
        arm.move_cartesian_traj([release_pose], speed=self.full_speed)
        arm.sync()
        
        arm.open_gripper()
        # smooth_pose = self.get_transform(y=-sgn*0.1, frame='world') * release_pose
        smooth_pose = release_pose * self.get_transform(a=sgn*0.3, frame='robot')
        smooth_pose.translation[2] = 0.07

        smooth_pose2 = self.get_transform(y=sgn*0.1, frame='world') * smooth_pose
        arm.move_cartesian_traj([smooth_pose, smooth_pose2], speed=self.full_speed, zone='z20')
        arm.sync()

    def fling(self, pick1: RigidTransform = None, pick2: RigidTransform = None, fling_to_fold=False, **kwargs):
        if pick1 is not None:
            pick_left, pick_right = self.assign_to_arm(pick1, pick2)
            pick_left, pick_right, left_sign, right_sign, left_place_diff, right_place_diff = self.optimize_angle_a_for_reachability(pick_left, pick_right)
        else:
            pick_left, pick_right = kwargs['pick_left'], kwargs['pick_right']
            left_sign, right_sign = kwargs['left_sign'], kwargs['right_sign']
            left_place_diff, right_place_diff = kwargs['left_place_diff'], kwargs['right_place_diff']

        center = pick_left.interpolate_with(pick_right, t=0.5)
        distance = np.linalg.norm(pick_left.translation - pick_right.translation)
        distance = max(min(distance, 0.3), 0.14) # [m]
        
        # 1. Move to grasp points
        offset_left = 0.007 if pick_left.translation[2] > 0.05 else 0.0
        offset_right = 0.007 if pick_right.translation[2] > 0.05 else 0.0

        approach_vector = self.get_transform(z=-0.04, frame='l_tcp')
        pick_left_top = pick_left * approach_vector
        self.y.left.goto_pose(pick_left_top, speed=self.full_speed, cf6=-1)

        approach_vector = self.get_transform(z=-0.04, frame='r_tcp')
        pick_right_top = pick_right * approach_vector
        self.y.right.goto_pose(pick_right_top, speed=self.full_speed)

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('fling', 'move to grasp point', poses=(pick1, pick2))

        pl = pick_left * self.get_transform(z=-0.003 + offset_left, b=left_sign*0.1, frame='l_tcp')
        pl2 = pl * self.get_transform(z=-0.001 + offset_left, x=left_sign*self.in_grasp_distance, b=left_sign*0.15, frame='l_tcp')

        pr = pick_right * self.get_transform(z=-0.003 + offset_right, b=right_sign*0.1, frame='r_tcp')
        pr2 = pr * self.get_transform(z=-0.001 + offset_right, x=right_sign*self.in_grasp_distance, b=right_sign*0.15, frame='r_tcp')
        
        # TODO: Currently trying this motion without planner, see how it goes...
        # l_path, r_path = self.model.plan_to_pose(self.y.left.get_joints(), self.y.right.get_joints(), self.model, pl, pr)
        # l_path2, r_path2 = self.model.plan_to_pose(l_path[-1], r_path[-1], self.model, pl2, pr2)
        # self.y.move_joints_sync(l_path, r_path, speed=self.half_speed)
        # self.y.move_joints_sync(l_path2, r_path2, speed=self.half_speed)

        self.y.move_cartesian_sync([pl, pl2], [pr, pr2], speed=self.half_speed)

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('fling', 'grasping', poses=(pick1, pick2))

        self.y.left.close_gripper()
        self.y.right.close_gripper()

        # 2. Move up and hold shirt
        motion_left, motion_right = [], []
        if pl2.translation[1] < pr2.translation[1] + 0.05: # Arms are crossed
            center = self.get_transform(z=0.02, frame='world') * pick_left.interpolate_with(pick_right, t=0.5)
            pl2.translation = (self.get_transform(y=distance/2, frame='world') * center).translation
            pr2.translation = (self.get_transform(y=-distance/2, frame='world') * center).translation
            motion_left.append(pl2)
            motion_right.append(pr2)

        # print(left_place_diff, right_place_diff)

        left_bsgn = 1.0 if left_place_diff == 0.0 else -1.0
        right_bsgn = 1.0 if right_place_diff == 0.0 else -1.0

        c_left = np.pi + left_place_diff
        c_right = -np.pi - right_place_diff

        motion_left.append(self.get_transform(x=0.45, y=distance/2, z=0.35, a=np.pi, b=0.0, c=c_left, from_frame='robot'))
        motion_right.append(self.get_transform(x=0.45, y=-distance/2, z=0.35, a=np.pi, b=0.0, c=c_right, from_frame='robot'))

        motion_left.append(self.get_transform(x=0.50, y=distance/2, z=0.5, a=np.pi, b=left_bsgn*0.2, c=c_left, from_frame='robot'))
        motion_right.append(self.get_transform(x=0.50, y=-distance/2, z=0.5, a=np.pi, b=right_bsgn*0.2, c=c_right, from_frame='robot'))

        self.y.move_cartesian_sync(motion_left, motion_right, speed=self.full_speed, zone='z20')
        time.sleep(0.1)

        # Stretch the shirt
        ymax = min(distance/2 + 0.25, 0.35)
        # ymax = distance/2 + 0.01 # For evaluating without stretching

        ml = self.get_transform(x=0.50, y=ymax, z=0.5, a=np.pi, b=left_bsgn*0.2, c=c_left, from_frame='robot')
        mr = self.get_transform(x=0.50, y=-ymax, z=0.5, a=np.pi, b=right_bsgn*0.2, c=c_right, from_frame='robot')
        self.y.left.move_contact(ml, speed=self.stretch_speed, max_torque=0.025)
        self.y.right.move_contact(mr, speed=self.stretch_speed, max_torque=0.025)
        if self.y.left.sync(2*self.TIMEOUT) or self.y.right.sync(2*self.TIMEOUT):
            self.exit_experiment('fling', 'stretching', poses=(pick1, pick2))

        pos_left = self.y.left.get_pose().translation
        pos_right = self.y.right.get_pose().translation

        collision_detected = np.linalg.norm(ml.translation - pos_left) > 0.005 or np.linalg.norm(mr.translation - pos_right) > 0.005
        if not collision_detected:
            ml = self.get_transform(x=0.50, y=0.12, z=0.5, a=np.pi, b=left_bsgn*0.2, c=c_left, from_frame='robot')
            mr = self.get_transform(x=0.50, y=-0.12, z=0.5, a=np.pi, b=right_bsgn*0.2, c=c_right, from_frame='robot')

            pos_left = ml.translation
            pos_right = mr.translation

            self.y.move_cartesian_sync([ml], [mr], speed=self.full_speed)
            if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
                self.exit_experiment('fling', 'no collision detected', poses=(pick1, pick2))

        # 3. Do the fling
        if fling_to_fold:
            z_bias_left, z_bias_right = 0.0, 0.0
            if np.array_equal(np.round(pick1.translation, 5), np.round(pick_left.translation, 5).astype(np.float32)):
                z_bias_left = 0.05
            else:
                z_bias_right = 0.05

            motion_left, motion_right = [], []
            motion_left.append(self.get_transform(x=0.76, y=pos_left[1] + 0.05, z=0.55 + z_bias_left, a=np.pi, b=left_bsgn*1.5, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.76, y=pos_right[1] + 0.05, z=0.55 + z_bias_right, a=np.pi, b=right_bsgn*1.5, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.52, y=pos_left[1] + 0.05, z=0.33 + z_bias_left, a=np.pi, b=left_bsgn*1.1, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.52, y=pos_right[1] + 0.05, z=0.33 + z_bias_right, a=np.pi, b=right_bsgn*1.1, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.40, y=pos_left[1] + 0.05, z=0.1 + z_bias_left, a=np.pi, b=left_bsgn*0.8, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.40, y=pos_right[1] + 0.05, z=0.1 + z_bias_right, a=np.pi, b=right_bsgn*0.8, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.2 + z_bias_right, y=pos_left[1] + 0.05, z=0.07 + z_bias_left, a=np.pi, b=left_bsgn*0.0, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.2 + z_bias_left, y=pos_right[1] + 0.05, z=0.07 + z_bias_right, a=np.pi, b=right_bsgn*0.0, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.1 + z_bias_right, y=pos_left[1] + 0.05, z=0.07 + z_bias_left, a=np.pi, b=left_bsgn*0.0, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.1 + z_bias_left, y=pos_right[1] + 0.05, z=0.07 + z_bias_right, a=np.pi, b=right_bsgn*0.0, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.2, y=pos_left[1] + 0.05, z=0.12 + z_bias_left, a=np.pi, b=left_bsgn*0.0, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.2, y=pos_right[1] + 0.05, z=0.12 + z_bias_right, a=np.pi, b=right_bsgn*0.0, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.27 + z_bias_left, y=pos_left[1] + 0.05, z=0.13 + z_bias_left, a=np.pi, b=-left_bsgn*0.8, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.27 + z_bias_right, y=pos_right[1] + 0.05, z=0.13 + z_bias_right, a=np.pi, b=-right_bsgn*0.8, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.33 + z_bias_left, y=pos_left[1] + 0.05, z=0.07, a=np.pi, b=-left_bsgn*1.1, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.33 + z_bias_right, y=pos_right[1] + 0.05, z=0.07, a=np.pi, b=-right_bsgn*1.1, c=c_right, from_frame='robot'))

            self.y.move_cartesian_sync(motion_left, motion_right, speed=self.fling_speed, zone='z50')
            if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
                self.exit_experiment('fling-to-fold', 'fling-to-fold motion', poses=(pick1, pick2))

            self.y.left.open_gripper()
            self.y.right.open_gripper()

            if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
                self.exit_experiment('fling-to-fold', 'open gripper', poses=(pick1, pick2))

            motion_left, motion_right = [], []
            motion_left.append(self.get_transform(x=0.47, y=pos_left[1], z=0.2, a=np.pi, b=-left_bsgn*0.6, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.47, y=pos_right[1], z=0.2, a=np.pi, b=-right_bsgn*0.6, c=c_right, from_frame='robot'))

            self.y.move_cartesian_sync(motion_left, motion_right, speed=self.fling_speed, zone='z50')
            if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
                self.exit_experiment('fling-to-fold', 'fling-to-fold exit', poses=(pick1, pick2))
            
        else:
            motion_left, motion_right = [], []
            motion_left.append(self.get_transform(x=0.76, y=pos_left[1], z=0.6, a=np.pi, b=left_bsgn*1.5, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.76, y=pos_right[1], z=0.6, a=np.pi, b=right_bsgn*1.5, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.52, y=pos_left[1], z=0.38, a=np.pi, b=left_bsgn*1.1, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.52, y=pos_right[1], z=0.38, a=np.pi, b=right_bsgn*1.1, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.30, y=pos_left[1], z=0.07, a=np.pi, b=left_bsgn*0.4, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.30, y=pos_right[1], z=0.07, a=np.pi, b=right_bsgn*0.4, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.26, y=pos_left[1], z=0.07, a=np.pi, b=left_bsgn*0.4, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.26, y=pos_right[1], z=0.07, a=np.pi, b=right_bsgn*0.4, c=c_right, from_frame='robot'))

            motion_left.append(self.get_transform(x=0.15, y=pos_left[1], z=0.07, a=np.pi, b=left_bsgn*0.0, c=c_left, from_frame='robot'))
            motion_right.append(self.get_transform(x=0.15, y=pos_right[1], z=0.07, a=np.pi, b=right_bsgn*0.0, c=c_right, from_frame='robot'))

            self.y.move_cartesian_sync(motion_left, motion_right, speed=[self.fling_speed, self.fling_speed, self.fling_speed, self.drag_speed], zone='z50')
            if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
                self.exit_experiment('fling', 'fling motion', poses=(pick1, pick2))

            self.y.left.open_gripper()
            self.y.right.open_gripper()

            if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
                self.exit_experiment('fling', 'open gripper', poses=(pick1, pick2))
    
    def pick_and_place(self, pick1: RigidTransform, pick2: RigidTransform, place1: RigidTransform, place2: RigidTransform, approach_height=0.1):
        (pick_left, place_left), (pick_right, place_right) = self.assign_to_arm((pick1, place1), (pick2, place2))

        # TODO(lars) fix place orientation, as it depends on the way between pick-and-place and is this even possible with our current yumi control?
        pick_left, pick_right, left_sign, right_sign, _, _ = self.optimize_angle_a_for_reachability(pick_left, pick_right, relative_direction_matters=False)
        place_left, place_right, _, _, _, _ = self.optimize_angle_a_for_reachability(place_left, place_right, relative_direction_matters=False)

        approach_vector_left = self.get_transform(z=-0.06, frame='l_tcp')
        approach_vector_right = self.get_transform(z=-0.06, frame='r_tcp')

        offset_left = 0.008 if pick_left.translation[2] > 0.05 else 0.0
        offset_right = 0.008 if pick_right.translation[2] > 0.05 else 0.0

        pick_left_top = pick_left * approach_vector_left
        pick_right_top = pick_right * approach_vector_right
        self.y.left.goto_pose(pick_left_top, speed=self.full_speed, cf6=-1)
        self.y.right.goto_pose(pick_right_top, speed=self.full_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-place', 'move to top pick points', poses=(pick1, pick2, place1, place2))

        pl = pick_left * self.get_transform(z=-0.002 + offset_left, b=left_sign*0.1, frame='l_tcp')
        pl2 = pl * self.get_transform(z=-0.001, x=left_sign*self.in_grasp_distance, b=left_sign*0.15, frame='l_tcp')
        self.y.left.move_cartesian_traj([pl, pl2], speed=self.half_speed)

        pr = pick_right * self.get_transform(z=-0.002 + offset_right, b=right_sign*0.1, frame='r_tcp')
        pr2 = pr * self.get_transform(z=-0.001, x=right_sign*self.in_grasp_distance, b=right_sign*0.15, frame='r_tcp')
        self.y.right.move_cartesian_traj([pr, pr2], speed=self.half_speed)

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-place', 'move to pick point', poses=(pick1, pick2, place1, place2))

        self.y.left.close_gripper()
        self.y.right.close_gripper()

        time.sleep(0.4)  # [s]
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-place', 'close gripper', poses=(pick1, pick2, place1, place2))

        approach_vector_left = self.get_transform(z=-approach_height, frame='l_tcp')
        approach_vector_right = self.get_transform(z=-approach_height, frame='r_tcp')

        place_left_top = place_left * approach_vector_left
        place_right_top = place_right * approach_vector_right

        half_approach_vector_left = self.get_transform(z=-approach_height/2, frame='l_tcp')
        half_approach_vector_right = self.get_transform(z=-approach_height/2, frame='r_tcp')

        pick_left_half_top = pick_left * half_approach_vector_left
        place_left_half_top = place_left * half_approach_vector_left
        pick_right_half_top = pick_right * half_approach_vector_right
        place_right_half_top = place_right * half_approach_vector_right

        self.y.move_cartesian_sync([pick_left_half_top, place_left_top, place_left_half_top], [pick_right_half_top, place_right_top, place_right_half_top], speed=self.full_speed, zone="z20")
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-place', 'move to place point', poses=(pick1, pick2, place1, place2))

        self.y.left.open_gripper()
        self.y.right.open_gripper()

        approach_vector_left = self.get_transform(z=-0.1, frame='l_tcp')
        approach_vector_right = self.get_transform(z=-0.1, frame='r_tcp')

        place_left_top = place_left * approach_vector_left
        place_right_top = place_right * approach_vector_right

        self.y.move_cartesian_sync([place_left_top], [place_right_top], speed=self.full_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-place', 'open gripper', poses=(pick1, pick2, place1, place2))

    def pick_and_hold(self, pick: RigidTransform = None, place: RigidTransform = None, hold: RigidTransform = None, **kwargs):
        if pick is not None:
            data_left, data_right = self.assign_to_arm((pick, place), hold)
            pick_with_left_arm = isinstance(data_left, tuple)

            pick, place = data_left if pick_with_left_arm else data_right
            hold = data_right if pick_with_left_arm else data_left

            pick_sign = 1.0
            if hold:
                if pick_with_left_arm:
                    pick, hold, pick_sign, _, _, _ = self.optimize_angle_a_for_reachability(pick, hold)
                else:
                    hold, pick, _, pick_sign, _, _ = self.optimize_angle_a_for_reachability(hold, pick)
            else:
                pick, pick_sign = self.optimize_angle_c(pick, frame=pick.from_frame)

            if pick_sign < 0.0:
                place = place * self.get_transform(c=np.pi, frame=place.from_frame)

        else:
            pick_with_left_arm = ('pick_left' in kwargs)
            pick_sign = kwargs['pick_sign']

            if pick_with_left_arm:
                pick, place = kwargs['pick_left'], kwargs['place_left']
                hold = kwargs['hold_right']
            else:
                pick, place = kwargs['pick_right'], kwargs['place_right']
                hold = kwargs['hold_left']

        pick_frame = 'l_tcp' if pick_with_left_arm else 'r_tcp'
        hold_frame = 'r_tcp' if pick_with_left_arm else 'l_tcp'

        arm_pick = self.y.left if pick_with_left_arm else self.y.right
        arm_hold = self.y.right if pick_with_left_arm else self.y.left

        if hold:
            arm_hold.close_gripper()

        # 1. Move to grasp points
        approach_vector = self.get_transform(z=-0.04, frame=pick_frame)
        approach_half_vector = self.get_transform(z=-0.02, frame=pick_frame)

        pick_top = pick * approach_vector
        pick_top_half = pick * approach_half_vector
        place_top_half = place * approach_half_vector

        arm_pick.goto_pose(pick_top, speed=self.full_speed)
        pl = pick * self.get_transform(z=-0.003, b=pick_sign*0.1, frame=pick_frame)
        pl2 = pl * self.get_transform(x=pick_sign*self.in_grasp_distance, z=-0.001, b=pick_sign*0.12, frame=pick_frame)
        arm_pick.move_cartesian_traj([pl, pl2], speed=self.half_speed)
        
        if hold:
            hold_top = hold * self.get_transform(z=-0.04, frame=hold_frame)
            hold_down = hold * self.get_transform(z=0.003, frame=hold_frame)
            hold_sign = -1 if pick_with_left_arm else 1
            hold_side = hold_down * self.get_transform(a=hold_sign * 0.5, frame=hold_frame)
            arm_hold.goto_pose(hold_top, speed=self.full_speed)
            arm_hold.move_cartesian_traj([hold_side], speed=self.half_speed)
        
        if arm_pick.sync(self.TIMEOUT) or arm_hold.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-hold', 'move to pick and hold pose', poses=(pick, place, hold))

        arm_pick.close_gripper()
        time.sleep(0.4)  # [s]
        if arm_pick.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-hold', 'close gripper', poses=(pick, place, hold))

        center = pick_top_half.interpolate_with(place_top_half, t=0.5)
        center = center * self.get_transform(z=-0.03, frame=pick_frame)

        arm_pick.move_cartesian_traj([pick_top_half], speed=self.move_clothes_speed)
        arm_pick.move_contact(place_top_half, speed=self.move_clothes_speed, max_torque=0.04)

        if arm_pick.sync(self.TIMEOUT) or arm_hold.sync(self.TIMEOUT):
            self.exit_experiment('pick-and-hold', 'move to place pose', poses=(pick, place, hold))

        arm_pick.open_gripper()
        arm_hold.open_gripper()

    def move(self, pick1: RigidTransform, pick2: RigidTransform, place1: RigidTransform, place2: RigidTransform):
        # Calculate place points
        if not place1:
            pass

        (pick_left, place_left), (pick_right, place_right) = self.assign_to_arm((pick1, place1), (pick2, place2))

        left_sign, right_sign = 1.0, 1.0
        pick_left, pick_right, left_sign, right_sign, _, _ = self.optimize_angle_a_for_reachability(pick_left, pick_right, relative_direction_matters=False)
        place_left, place_right, _, _, _, _ = self.optimize_angle_a_for_reachability(place_left, place_right, relative_direction_matters=False)

        approach_vector = self.get_transform(z=-0.04, frame='l_tcp')

        pick_left_top = pick_left * approach_vector
        approach_vector = self.get_transform(z=-0.04, frame='r_tcp')
        pick_right_top = pick_right * approach_vector
        self.y.move_cartesian_sync([pick_left_top], [pick_right_top], speed=self.full_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('move', 'move to top grasp pose', poses=(pick1, pick2, place1, place2))

        pl = pick_left * self.get_transform(z=-0.004, b=-left_sign*0.1, frame='l_tcp')
        pl2 = pl * self.get_transform(z=-0.001, x=-left_sign*self.in_grasp_distance, b=-left_sign*0.15, frame='l_tcp')
        self.y.left.move_cartesian_traj([pl, pl2], speed=self.half_speed)

        pr = pick_right * self.get_transform(z=-0.004, b=-right_sign*0.1, frame='r_tcp')
        pr2 = pr * self.get_transform(z=-0.001, x=-right_sign*self.in_grasp_distance, b=-right_sign*0.15, frame='r_tcp')
        self.y.right.move_cartesian_traj([pr, pr2], speed=self.half_speed)

        self.y.left.close_gripper()
        self.y.right.close_gripper()

        approach_vector = self.get_transform(z=-0.02, frame='l_tcp')
        pick_left_top = pick_left * approach_vector
        place_left_top = place_left * approach_vector
        approach_vector = self.get_transform(z=-0.02, frame='r_tcp')
        pick_right_top = pick_right * approach_vector
        place_right_top = place_right * approach_vector

        self.y.move_cartesian_sync([pick_left_top, place_left_top, place_left], [pick_right_top, place_right_top, place_right], speed=self.full_speed, zone='z15')
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('move', 'move to place pose', poses=(pick1, pick2, place1, place2))

        self.y.left.open_gripper()
        self.y.right.open_gripper()

        approach_vector = self.get_transform(z=-0.04, frame='l_tcp')
        place_left_top = place_left * approach_vector
        approach_vector = self.get_transform(z=-0.04, frame='r_tcp')
        place_right_top = place_right * approach_vector
        self.y.move_cartesian_sync([place_left_top], [place_right_top], speed=self.full_speed, zone='z15')
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('move', 'move up motion', poses=(pick1, pick2, place1, place2))

    def move_to_center(self, image: RgbdImage = None, **kwargs):
        image, _ = (image, None) if image else self.take_image()

        pick1, pick2, place1, place2 = self.mh.calculate(image.color.raw_data, **kwargs)

        points_3d = self.get_pointcloud(image)
        pick1 = self.pixel_to_transform(pick1, image.shape, points_3d)
        pick2 = self.pixel_to_transform(pick2, image.shape, points_3d)
        place1 = self.pixel_to_transform(place1, image.shape, points_3d)
        place2 = self.pixel_to_transform(place2, image.shape, points_3d)

        if place1.translation[2] < 0.047:
            place1.translation[2] = 0.047

        if place2.translation[2] < 0.047:
            place2.translation[2] = 0.047

        assert self.is_pose_within_workspace(pick1) and self.is_pose_within_workspace(pick2), 'pose is not safe'

        self.move(pick1, pick2, place1, place2)

    def drag(self, pick1: RigidTransform, pick2: RigidTransform, place1: RigidTransform, place2: RigidTransform, sgn: int):
        # Calculate place points
        if not place1:
            pass

        (pick_left, place_left), (pick_right, place_right) = self.assign_to_arm((pick1, place1), (pick2, place2))

        left_sign, right_sign = sgn, sgn # Z1.0, 1.0
        pick_left, pick_right, left_sign, right_sign, _, _ = self.optimize_angle_a_for_reachability(pick_left, pick_right)
        place_left, place_right, _, _, _, _ = self.optimize_angle_a_for_reachability(place_left, place_right)

        approach_vector = self.get_transform(z=-0.04, frame='l_tcp')

        pick_left_top = pick_left * approach_vector
        approach_vector = self.get_transform(z=-0.04, frame='r_tcp')
        pick_right_top = pick_right * approach_vector
        self.y.move_cartesian_sync([pick_left_top], [pick_right_top], speed=self.full_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('drag', 'move to top grasp pose', poses=(pick1, pick2, place1, place2))

        pl = pick_left * self.get_transform(z=-0.004, b=left_sign*0.1, frame='l_tcp')
        pl2 = pl * self.get_transform(z=-0.001, x=left_sign*self.in_grasp_distance, b=left_sign*0.15, frame='l_tcp')
        self.y.left.move_cartesian_traj([pl, pl2], speed=self.half_speed)

        pr = pick_right * self.get_transform(z=-0.004, b=right_sign*0.1, frame='r_tcp')
        pr2 = pr * self.get_transform(z=-0.001, x=right_sign*self.in_grasp_distance, b=right_sign*0.15, frame='r_tcp')
        self.y.right.move_cartesian_traj([pr, pr2], speed=self.half_speed)

        self.y.left.close_gripper()
        self.y.right.close_gripper()

        approach_vector = self.get_transform(z=-0.02, frame='l_tcp')
        pick_left_top = pick_left * approach_vector
        place_left_top = place_left * approach_vector
        approach_vector = self.get_transform(z=-0.02, frame='r_tcp')
        pick_right_top = pick_right * approach_vector
        place_right_top = place_right * approach_vector

        self.y.move_cartesian_sync([pick_left_top, place_left_top, place_left], [pick_right_top, place_right_top, place_right], speed=self.full_speed, zone='z15')
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('drag', 'move to place pose', poses=(pick1, pick2, place1, place2))

        self.y.left.open_gripper()
        self.y.right.open_gripper()

        approach_vector = self.get_transform(z=-0.04, frame='l_tcp')
        place_left_top = place_left * approach_vector
        approach_vector = self.get_transform(z=-0.04, frame='r_tcp')
        place_right_top = place_right * approach_vector
        self.y.move_cartesian_sync([place_left_top], [place_right_top], speed=self.full_speed, zone='z15')
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('drag', 'move up motion', poses=(pick1, pick2, place1, place2))

    def final_fling(self, pick1: RigidTransform, pick2: RigidTransform = None):
        pick_left, pick_right = self.assign_to_arm(pick1, pick2)
        pick_left, pick_right, left_sign, right_sign, left_place_diff, right_place_diff = self.optimize_angle_a_for_reachability(pick_left, pick_right)
       
        distance = np.linalg.norm(pick_left.translation - pick_right.translation)
        
        # 1. Move to grasp points
        offset_left = 0.007 if pick_left.translation[2] > 0.05 else 0.0
        offset_right = 0.007 if pick_right.translation[2] > 0.05 else 0.0

        approach_vector = self.get_transform(z=-0.04, frame='l_tcp')
        pick_left_top = pick_left * approach_vector
        self.y.left.goto_pose(pick_left_top, speed=self.full_speed, cf6=-1)

        approach_vector = self.get_transform(z=-0.04, frame='r_tcp')
        pick_right_top = pick_right * approach_vector
        self.y.right.goto_pose(pick_right_top, speed=self.full_speed)

        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('final-fling', 'move to grasp point', poses=(pick1, pick2))

        pl = pick_left * self.get_transform(z=-0.003 + offset_left, b=left_sign*0.1, frame='l_tcp')
        pl2 = pl * self.get_transform(z=-0.001 + offset_left, x=left_sign*self.in_grasp_distance, b=left_sign*0.15, frame='l_tcp')

        pr = pick_right * self.get_transform(z=-0.003 + offset_right, b=right_sign*0.1, frame='r_tcp')
        pr2 = pr * self.get_transform(z=-0.001 + offset_right, x=right_sign*self.in_grasp_distance, b=right_sign*0.15, frame='r_tcp')

        self.y.move_cartesian_sync([pl, pl2], [pr, pr2], speed=self.half_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('final-fling', 'grasping', poses=(pick1, pick2))

        self.y.left.close_gripper()
        self.y.right.close_gripper()

        time.sleep(0.2)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('final-fling', 'grasping', poses=(pick1, pick2))

        # 2. Move up and hold shirt
        motion_left, motion_right = [], []
        # if pl2.translation[1] < pr2.translation[1] + 0.05: # Arms are crossed
        #     center = self.get_transform(z=0.02, frame='world') * pick_left.interpolate_with(pick_right, t=0.5)
        #     pl2.translation = (self.get_transform(y=distance/2, frame='world') * center).translation
        #     pr2.translation = (self.get_transform(y=-distance/2, frame='world') * center).translation
        #     motion_left.append(pl2)
        #     motion_right.append(pr2)

        left_bsgn = 1.0 if left_place_diff == 0.0 else -1.0
        right_bsgn = 1.0 if right_place_diff == 0.0 else -1.0

        c_left = np.pi + left_place_diff
        c_right = -np.pi - right_place_diff

        fling_pose_left = self.get_transform(x=0.5, y=-0.20 + distance/2, z=0.32, a=np.pi, b=left_bsgn*0.2, c=c_left, from_frame='robot')
        target_pose_left = self.get_transform(x=0.5, y=-0.20 + distance/2, z=0.07, a=np.pi, b=left_bsgn*0.4, c=c_left, from_frame='robot')

        fling_pose_right = self.get_transform(x=0.5, y=-0.20 - distance/2, z=0.32, a=np.pi, b=right_bsgn*0.2, c=c_right, from_frame='robot')
        target_pose_right = self.get_transform(x=0.5, y=-0.20 - distance/2, z=0.07, a=np.pi, b=right_bsgn*0.4, c=c_right, from_frame='robot')

        drag_distance = 0.04

        motion_left.append(fling_pose_left)
        motion_right.append(fling_pose_right)

        self.y.move_cartesian_sync(motion_left, motion_right, speed=self.full_speed, zone='z20')
        time.sleep(0.1)

        # 3. Do the fling
        motion_left, motion_right = [], []
        motion_left.append(self.get_transform(x=0.24, z=0.28, frame='world') * target_pose_left * self.get_transform(b=-1.1, c=0.0, from_frame='l_tcp', to_frame='robot'))
        motion_left.append(self.get_transform(x=drag_distance, frame='world') * target_pose_left * self.get_transform(c=0.0, from_frame='l_tcp', to_frame='robot'))
        motion_left.append(target_pose_left * self.get_transform(c=0.0, from_frame='l_tcp', to_frame='robot'))

        motion_right.append(self.get_transform(x=0.24, z=0.28, frame='world') * target_pose_right * self.get_transform(b=-1.1, c=0.0, from_frame='r_tcp', to_frame='robot'))
        motion_right.append(self.get_transform(x=drag_distance, frame='world') * target_pose_right * self.get_transform(c=0.0, from_frame='r_tcp', to_frame='robot'))
        motion_right.append(target_pose_right * self.get_transform(c=0.0, from_frame='r_tcp', to_frame='robot'))

        self.y.move_cartesian_sync(motion_left, motion_right, speed=[self.fling_speed, self.fling_speed, self.drag_speed], zone='z50')
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('final-fling', 'fling motion', poses=(pick1, pick2))

        self.y.left.open_gripper()
        self.y.right.open_gripper()

    def execute_action(self, action: Dict, image: RgbdImage):
        if action['type'] == 'done':
            self.home()
            return

        assert 'poses' in action, 'action should include list of poses'
        assert len(action['poses']) == 2, 'action/poses should include 2 poses'

        pixel1 = PlanarTransform.fromRelativePixelDictionary(action['poses'][0], image.shape)
        pixel2 = PlanarTransform.fromRelativePixelDictionary(action['poses'][1], image.shape)
        points_3d = self.get_pointcloud(image)
        pick1 = self.pixel_to_transform(pixel1, image.shape, points_3d)
        pick2 = self.pixel_to_transform(pixel2, image.shape, points_3d)

        # This is just to make double sure, should be catched by inference class way before
        assert self.is_pose_within_workspace(pick1) and self.is_pose_within_workspace(pick2), 'pose is not safe'

        if action['type'] == 'fling':
            if 'transforms' in action:
                self.fling(**action['transforms'])
            else:
                self.fling(pick1, pick2)
        
        if action['type'] == 'fling-to-fold':
            self.fling(pick1, pick2, fling_to_fold=True)

        elif action['type'] == 'move-to-center':
            self.move_to_center(image=image, x_only=True)
        
        elif action['type'] == 'pick-and-hold':
            place_pixel, hold_pixel = self.hph.refine_place_and_hold_points(pixel1, pixel2, image=image.color.raw_data)

            place = self.pixel_to_transform(place_pixel, image.shape, points_3d)
            hold = self.pixel_to_transform(hold_pixel, image.shape, points_3d) if hold_pixel is not None else None

            assert self.is_pose_within_workspace(place), 'pose is not safe'
            if hold_pixel is not None:
                assert self.is_pose_within_workspace(hold), 'pose is not safe'
            
            if 'transforms' in action:
                self.pick_and_hold(**action['transforms'])
            else:
                self.pick_and_hold(pick1, place, hold)
        
        elif action['type'] == 'drag':
            pose1, pose2 = action['poses']
            H, W, C = image.color.data.shape
            pnt1 = np.round((pose1['x']*W, pose1['y']*H)).astype(np.int64)
            pnt2 = np.round((pose2['x']*W, pose2['y']*H)).astype(np.int64)
            mask, info = segment(image.color.data)
            com = np.round((info['x']*W, info['y']*H)).astype(np.int64)
            points = [pnt1, pnt2]
            x_coords, y_coords = zip(*points)
            A = np.vstack([x_coords,np.ones(len(x_coords))]).T
            m, c = np.linalg.lstsq(A, y_coords)[0]

            axis = 0 if abs(pnt1[0] - pnt2[0]) < abs(pnt1[1] - pnt2[1]) else 1
            sgn = 1 if com[axis] < pnt1[axis] else -1

            step_size = 50
            def find_place_horizontal(pnt, W, H):
                pnt_dx = pnt[0] + step_size*sgn
                m_d = -1 / m
                c_d = pnt[1] - m_d * pnt[0]
                pnt_dy = pnt_dx * m_d + c_d
                return np.round((np.clip(pnt_dx,0,W), np.clip(pnt_dy,0,H))).astype(np.int64)

            def find_place_vertical(pnt, W, H):
                pnt_dy = pnt[1] + step_size*sgn
                m_d = -1 / m
                c_d = pnt[1] - m_d * pnt[0]
                pnt_dx = (pnt_dy - c_d) / m_d
                return np.round((np.clip(pnt_dx,0,W), np.clip(pnt_dy,0,H))).astype(np.int64)    

            if axis:
                pnt1_d = find_place_vertical(pnt1, W, H)
                pnt2_d = find_place_vertical(pnt2, W, H)
            else:
                pnt1_d = find_place_horizontal(pnt1, W, H)
                pnt2_d = find_place_horizontal(pnt2, W, H)
            
            # import ipdb; ipdb.set_trace()
            pixel3 = PlanarTransform.fromRelativePixelDictionary({'x': pnt1_d[0] / W, 'y': pnt1_d[1] / H, 'theta': pose1['theta']}, image.shape)
            pixel4 = PlanarTransform.fromRelativePixelDictionary({'x': pnt2_d[0] / W, 'y': pnt2_d[1] / H, 'theta': pose1['theta']}, image.shape)
            place1 = self.pixel_to_transform(pixel3, image.shape, points_3d)
            place2 = self.pixel_to_transform(pixel4, image.shape, points_3d)

            if place1.translation[2] < 0.05:
                place1.translation[2] = 0.05

            if place2.translation[2] < 0.05:
                place2.translation[2] = 0.05

            self.drag(pick1, pick2, place1, place2, sgn)

        self.home()

    def execute_fold(self, instruction: Instruction, image: RgbdImage, evaluation, final_fling=False):
        for i, l in enumerate(instruction.folding_lines):
            if self.mh.should_move_for_folding(image.color.raw_data, center_threshold=0.65, top_threshold_px=80):
                with evaluation.profile('move-fold'):
                    self.move_to_center(image=image, x_only=True, x_offset=-30)

                    for j in range(i, len(instruction.folding_lines)):
                        instruction.folding_lines[j][0] += self.mh.current_transform
                        instruction.folding_lines[j][1] += self.mh.current_transform

                    if instruction.grasp_points is not None:
                        instruction.grasp_points[0] += self.mh.current_transform
                        instruction.grasp_points[1] += self.mh.current_transform
                    
                    self.home()

                with evaluation.profile('camera'):
                    image, _ = self.take_image()

            with evaluation.profile('pick-and-place'):
                is_last = (i == len(instruction.folding_lines) - 1)
                line = Line(start=[l[0][0] / image.shape[1], l[0][1] / image.shape[0]], end=[l[1][0] / image.shape[1], l[1][1] / image.shape[0]])

                pick1, pick2, place1, place2 = self.fh.calculate(image.color.raw_data, line, save=True)

                points_3d = self.get_pointcloud(image)
                pick1 = self.pixel_to_transform(pick1, image.shape, points_3d)
                pick2 = self.pixel_to_transform(pick2, image.shape, points_3d)
                place1 = self.pixel_to_transform(place1, image.shape, points_3d)
                place2 = self.pixel_to_transform(place2, image.shape, points_3d)

                self.pick_and_place(pick1, pick2, place1, place2, approach_height=0.04 + i * 0.03)
                self.home()

            if not is_last:
                with evaluation.profile('camera'):
                    image, _ = self.take_image()

        if final_fling and instruction.grasp_points is not None:
            with evaluation.profile('camera'):
                image, _ = self.take_image()

            with evaluation.profile('final-fling'):
                pixel1 = self.tm.find_nearest_grasp_pose(image.color.raw_data, instruction.grasp_points[0])
                pixel2 = self.tm.find_nearest_grasp_pose(image.color.raw_data, instruction.grasp_points[1])
                points_3d = self.get_pointcloud(image)

                pick1 = self.pixel_to_transform(pixel1, image.shape, points_3d)
                pick2 = self.pixel_to_transform(pixel2, image.shape, points_3d)

                # Just some template as some methods are not single arm only, won't be executed
                # pick2 = Experiment.get_transform(x=0.5, y=1.0, z=0.05, a=np.pi, b=0.0, c=np.pi, from_frame='r_tcp')

                self.final_fling(pick1, pick2)
                self.home()

    def execute_fold_after_f2f(self, inf, image: RgbdImage, evaluation):
        from selection import Top
        selection = Top(5)

        with evaluation.profile('predict'):
            action = inf.predict_action(image, selection, action_type='fling-to-fold', save=True)
        
        with evaluation.profile('pick-and-place'):
            pixel1 = PlanarTransform.fromRelativePixelDictionary(action['poses'][0], image.shape)
            pixel2 = PlanarTransform.fromRelativePixelDictionary(action['poses'][1], image.shape)
            points_3d = self.get_pointcloud(image)
            pick1 = self.pixel_to_transform(pixel1, image.shape, points_3d)
            pick2 = self.pixel_to_transform(pixel2, image.shape, points_3d)

            left = pick1 if pick1.translation[1] > pick2.translation[1] else pick2
            right = pick2 if pick1.translation[1] > pick2.translation[1] else pick1
            sleeve_dir = 1.0
            right_sleeve = False
            if left.translation[0] > right.translation[0]:
                sleeve_dir = -1.0
                right_sleeve = True

            transform_bottom = self.get_transform(x=-0.15, y=-sleeve_dir*0.02, frame='world')
            transform_sleeve = self.get_transform(x=-0.27, y=-sleeve_dir*0.02, frame='world')

            if pick1.translation[0] < pick2.translation[0]:
                place1 = transform_bottom * pick1
                place2 = transform_sleeve * pick2
            else:
                place1 = transform_sleeve * pick1
                place2 = transform_bottom * pick2

            self.pick_and_place(pick1, pick2, place1, place2)
            self.home()

        with evaluation.profile('camera'):
            image, _ = self.take_image()

        with evaluation.profile('pick-and-place'):
            x_val = ((pixel1.position[0] + pixel2.position[0] + sleeve_dir * 80) / 2) / image.color.shape[1]
            y_start, y_end = 0.99, 0.01
            if not right_sleeve:
                tmp = y_start
                y_start = y_end
                y_end = tmp
            line = Line(start=[x_val, y_start], end=[x_val, y_end])

            fh = FoldingHeuristic()
            pick_left, pick_right, place_left, place_right = fh.calculate(image.color.raw_data, line, save=True)

            points_3d = self.get_pointcloud(image)
            pick1 = self.pixel_to_transform(pick_left, image.shape, points_3d)
            pick2 = self.pixel_to_transform(pick_right, image.shape, points_3d)
            place1 = self.pixel_to_transform(place_left, image.shape, points_3d)
            place2 = self.pixel_to_transform(place_right, image.shape, points_3d)

            self.pick_and_place(pick1, pick2, place1, place2)
            self.home()

    def execute_2s_fold(self, image: RgbdImage, evaluation=None):
        # Template matching
        mask, _, contour = segment(image.color.data, return_contour=True)
        instruction = exp.tm.get_matched_instruction(mask, template_name='2s')

        start = time.time()

        if self.mh.should_move_for_folding(image.color.raw_data, center_threshold=0.62, top_threshold_px=80):
            self.move_to_center(image=image, x_only=True, x_offset=25, margin_bottom=260)

            for j in range(len(instruction.folding_lines)):
                instruction.folding_lines[j][0] += self.mh.current_transform
                instruction.folding_lines[j][1] += self.mh.current_transform
            
            self.home()
            image, _ = self.take_image()
            mask, _, contour = segment(image.color.data, return_contour=True)

        # Get pick points based on single line
        p1 = find_first_point_in_contour(instruction.folding_lines[0][0], instruction.folding_lines[0][1], contour)
        p2 = find_first_point_in_contour(instruction.folding_lines[0][1], instruction.folding_lines[0][0], contour)

        diff = p2 - p1
        theta = np.arctan2(diff[1], diff[0]) - np.pi / 2

        sgn_right = 1.0
        print(theta)
        if theta > -1.1:  # [rad]
            sgn_right *= -1.0

        points_3d = self.get_pointcloud(image)
        p1 = self.pixel_to_transform(p1, image.shape, points_3d)
        p2 = self.pixel_to_transform(p2, image.shape, points_3d)

        p1.from_frame = 'l_tcp'
        p2.from_frame = 'l_tcp'

        pick_left = p1 * self.get_transform(z=-0.001, c=theta, frame='l_tcp')
        regrasp_left = p2 * self.get_transform(z=-0.002, c=theta, frame='l_tcp')

        center_between_picks = pick_left.interpolate_with(regrasp_left, t=0.5)
        center_between_picks.from_frame = 'r_tcp'
        pick_right = center_between_picks * self.get_transform(y=-sgn_right*0.03, x=-0.026, z=0.014, c=np.pi, frame='r_tcp')
        # pick_right = self.get_transform(x=center_between_picks.translation[0] - 0.03, y=center_between_picks.translation[1] - 0.016, z=center_between_picks.translation[2] - 0.014, a=np.pi, b=0.0, c=-np.pi/2, from_frame='r_tcp')

        distance = np.linalg.norm(pick_left.translation - pick_right.translation)

        # 1. Move to grasp poses
        pick_left_top = pick_left * self.get_transform(x=-0.01, z=-0.04, frame='l_tcp')
        self.y.left.goto_pose(pick_left_top, speed=self.full_speed)
        
        pl = pick_left * self.get_transform(x=-0.01, z=0.001, b=-0.1, frame='l_tcp')
        pl2 = pl * self.get_transform(x=-0.002, b=-0.02, frame='l_tcp')
        self.y.left.move_cartesian_traj([pl, pl2], speed=self.half_speed)
        
        pick_right_top = pick_right * self.get_transform(z=-0.06, frame='r_tcp')
        self.y.right.move_cartesian_traj([pick_right_top, pick_right], speed=self.full_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('2s-fold', '')

        self.y.left.close_gripper()
        self.y.right.close_gripper()

        self.y.left.sync()
        self.y.right.sync()
        time.sleep(0.1)

        # 2. Avoid motion right, move up left
        pr = pick_right * self.get_transform(a=sgn_right*0.85, frame='r_tcp')
        self.y.right.goto_pose(pr, speed=self.full_speed)
        self.y.left.goto_pose(pick_left_top, speed=self.move_clothes_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('2s-fold', '')

        # 3. Move left over right
        regrasp_left_intermediate = self.get_transform(z=0.10, from_frame='world') * center_between_picks
        rl = regrasp_left * self.get_transform(x=0.01, a=0.2, frame='l_tcp')
        self.y.left.move_cartesian_traj([regrasp_left_intermediate, rl], speed=[self.full_speed, self.move_clothes_speed], zone='z20')
        self.y.left.sync()

        self.y.left.open_gripper()
        self.y.left.sync()

        # 4. Regrasp
        rl = regrasp_left * self.get_transform(x=0.04, b=0.1, frame='l_tcp')
        self.y.left.goto_pose(rl, speed=self.full_speed)
        self.y.left.sync()

        self.y.left.close_gripper()
        time.sleep(0.15)
        self.y.left.sync()
        time.sleep(0.1)

        # 5. Retract left arm
        # rl = self.get_transform(z=0.1, frame='world') * regrasp_left
        # rl2 = self.get_transform(x=0.597, y=-0.092, z=0.160, a=np.pi, b=0.0, c=3*np.pi/4, from_frame='l_tcp')
        # self.y.left.move_cartesian_traj([regrasp_left_intermediate], speed=self.full_speed, zone='z10')
        # self.y.left.sync()

        theta_left = np.pi/2 + 0.4
        theta_right = -np.pi/2

        # 6. Hang shirt in front
        motion_left, motion_right = [regrasp_left_intermediate], [pr]

        motion_left.append(self.get_transform(x=0.456, y=distance/2 - 0.035, z=0.10, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp'))
        motion_right.append(self.get_transform(x=0.466, y=-distance/2 + 0.035, z=0.10, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp'))

        motion_left.append(self.get_transform(x=0.400, y=distance/2 - 0.035, z=0.430, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp'))
        motion_right.append(self.get_transform(x=0.400, y=-distance/2 + 0.035, z=0.430, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp'))

        self.y.move_cartesian_sync(motion_left, motion_right, speed=[self.move_clothes_speed, self.move_clothes_speed, self.full_speed], zone='z10')
        if self.y.left.sync(2*self.TIMEOUT) or self.y.right.sync(2*self.TIMEOUT):
            self.exit_experiment('2s-fold', 'hanging')

        # Stretch the shirt
        # ml = self.get_transform(x=0.400, y=distance/2 - 0.032, z=0.430, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp')
        # mr = self.get_transform(x=0.400, y=-distance/2 + 0.032, z=0.430, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp')
        # self.y.left.move_contact(ml, speed=self.stretch_speed, max_torque=0.025)
        # self.y.right.move_contact(mr, speed=self.stretch_speed, max_torque=0.025)
        # if self.y.left.sync(2*self.TIMEOUT) or self.y.right.sync(2*self.TIMEOUT):
        #     self.exit_experiment('2s-fold', 'stretching', poses=(pick1, pick2))

        y_left = distance/2 - 0.035 # self.y.left.get_pose().translation[1]
        y_right = -distance/2 + 0.035 # self.y.right.get_pose().translation[1]

        # 7. Shaking
        motion_left, motion_right = [], []
        for _ in range(10):
            motion_left.append(self.get_transform(x=0.380, y=y_left - 0.01, z=0.460, a=np.pi+0.15, b=0.1, c=theta_left - 0.1, from_frame='l_tcp'))
            motion_right.append(self.get_transform(x=0.370, y=y_right - 0.01, z=0.460, a=np.pi-0.15, b=-0.1, c=theta_right + 0.1, from_frame='r_tcp'))

            motion_left.append(self.get_transform(x=0.400, y=y_left, z=0.430, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp'))
            motion_right.append(self.get_transform(x=0.400, y=y_right, z=0.430, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp'))

        self.y.move_cartesian_sync(motion_left, motion_right, speed=self.fling_speed, zone='z1')
        if self.y.left.sync(2*self.TIMEOUT) or self.y.right.sync(2*self.TIMEOUT):
            self.exit_experiment('2s-fold', 'shaking')

        # 8. Put it down
        motion3_left = self.get_transform(x=0.310, y=y_left, z=0.25, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp')
        motion3_right = self.get_transform(x=0.310, y=y_right, z=0.25, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp')

        motion4_left = self.get_transform(x=0.510, y=y_left, z=0.05, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp')
        motion4_right = self.get_transform(x=0.510, y=y_right, z=0.05, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp')

        motion5_left = self.get_transform(x=0.510, y=y_left, z=0.05, a=np.pi, b=0.0, c=theta_left, from_frame='l_tcp')
        motion5_right = self.get_transform(x=0.510, y=y_right, z=0.05, a=np.pi, b=0.0, c=theta_right, from_frame='r_tcp')

        self.y.move_cartesian_sync([motion3_left, motion4_left, motion5_left], [motion3_right, motion4_right, motion5_right], speed=self.fling_speed, zone='z20')
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('2s-fold', '')

        self.y.left.open_gripper()
        self.y.right.open_gripper()
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('2s-fold', '')

        # 9. Move up
        motion5_left = self.get_transform(z=0.06, frame='world') * motion5_left
        motion5_right = self.get_transform(z=0.06, frame='world') * motion5_right

        self.y.move_cartesian_sync([motion5_left], [motion5_right], speed=self.full_speed)
        if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
            self.exit_experiment('2s-fold', '')

        # 10. Smooth it
        # motion5_left1 = self.get_transform(x=0.500, y=0.03, z=0.10, a=np.pi - 0.5, b=0.0, c=np.pi/2, from_frame='l_tcp')
        # motion5_left2 = self.get_transform(x=0.500, y=0.03, z=0.065, a=np.pi - 1.0, b=0.0, c=np.pi/2, from_frame='l_tcp')
        # motion5_right1 = self.get_transform(x=0.430, y=-0.075, z=0.10, a=np.pi - 0.4, b=0.0, c=np.pi, from_frame='r_tcp')
        # motion5_right2 = self.get_transform(x=0.430, y=-0.075, z=0.051, a=np.pi - 0.8, b=0.0, c=np.pi, from_frame='r_tcp')
        # self.y.move_cartesian_sync([motion5_left1, motion5_left2], [motion5_right1, motion5_right2], speed=self.full_speed)
        # if self.y.left.sync(self.TIMEOUT) or self.y.right.sync(self.TIMEOUT):
        #     self.exit_experiment('2s-fold', '')

        # pl = self.get_transform(z=-0.02, frame='world') * motion5_left2
        # pl2 = self.get_transform(y=0.12, frame='world') * pl
        # pl3 = self.get_transform(y=0.02, z=0.05, frame='world') * pl2
        # self.y.left.move_cartesian_traj([pl, pl2, pl3], speed=self.half_speed, zone='z20')
        # self.y.left.sync()

        duration = time.time() - start
        print(f'duration: {duration:0.4f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--take-current-image', action='store_true', help='takes an image and saved it into data/current/image.png')
    parser.add_argument('--take-image-to-annotate', action='store_true', help='takes an image and uploads it to the database')
    parser.add_argument('--take-image-for-class', dest='take_image_for_class', type=str, default=None, help='takes an image and saves it (without using the database)')
    parser.add_argument('--take-coverage', action='store_true', help='takes an image and calculates the coverage')
    parser.add_argument('--number', dest='number', type=int, default=1, help='in combination with take-image-to-annotate, take multiple images with a few seconds sleep in between')
    parser.add_argument('--do-the-fling', action='store_true', help='executes the fling primitive')
    parser.add_argument('--do-fling-to-fold', action='store_true', help='executes the fling2fold primitive')
    parser.add_argument('--do-final-fling', action='store_true', help='executes a small final fling for a folded cloth')
    parser.add_argument('--do-f2f-fold', action='store_true', help='executes a second fold after fling-to-fold')
    parser.add_argument('--do-2s-fold', action='store_true', help='executes the 2 seconds fold heuristic')
    parser.add_argument('--do-pick-and-hold-primitive', action='store_true', help='executes a pick-and-hold primitive')
    parser.add_argument('--do-move-primitive', action='store_true', help='executes a move primitive')
    parser.add_argument('--do-fold-instruction', dest='instruction', type=str, help='executes folding for the given instruction, e.g. `shirt`')
    parser.add_argument('--execute-live', action='store_true', help='waits for actions to execute from the live tab in the annotation / folding database')
    parser.add_argument('--check-calibration', action='store_true')
    parser.add_argument('--check-reachability', action='store_true')
    parser.add_argument('--gen-rand-scene', action='store_true')
    parser.add_argument('--redo-action-on-exit', dest='exit', type=str, default=None, help='redo the action that cuases the exit with id. for debuggin')
    parser.add_argument('--eval-classifier', action='store_true')
    
    args = parser.parse_args()

    exp = Experiment(save_on_exit=False)
    exp.home()


    if args.take_current_image:
        exp.take_image()

    if args.take_image_to_annotate:
        for i in range(args.number):
            print('Take new image for annotation...')
            exp.take_image_for_annotation()
            time.sleep(3.5)

    if args.take_coverage:
        print(exp.take_coverage() / 0.240)

    if args.do_the_fling:
        # pick1 = self.get_transform(x=0.26, y=-0.09, z=0.055, a=np.pi, b=0.0, c=-np.pi/2+0.2, from_frame='l_tcp')
        pick1 = Experiment.get_transform(x=0.26, y=0.09, z=0.055, a=np.pi, b=0.0, c=-np.pi, from_frame='l_tcp')
        pick2 = Experiment.get_transform(x=0.47, y=-0.04, z=0.055, a=np.pi, b=0.0, c=np.pi, from_frame='r_tcp')

        exp.fling(pick1, pick2)
        exp.home()

    if args.do_fling_to_fold:
        # pick1 = Experiment.get_transform(x=0.26, y=-0.09, z=0.055, a=np.pi, b=0.0, c=-np.pi/2+0.2, from_frame='l_tcp')
        pick1 = Experiment.get_transform(x=0.31, y=-0.15, z=0.055, a=np.pi, b=-0.35, c=0, from_frame='l_tcp')
        pick2 = Experiment.get_transform(x=0.26, y=0.15, z=0.055, a=np.pi, b=-0.35, c=0, from_frame='r_tcp')

        exp.fling(pick1, pick2, fling_to_fold=True)
        exp.home()
    
    if args.do_f2f_fold:
        # image, image_before_ortho = exp.take_image()
        # exp.move_to_center(image=image, x_only=True)
        # exp.home()

        from inference import Inference
        from selection import Top
        inference = Inference(
            multi_model_name='multi-02092022-emb-f2f.pth',
            primitives_model_name='f2f_2022-02-09.pth',
            experiment=exp,
        )
        selection = Top(20)
        image, image_before_ortho = exp.take_image()
        action = inference.predict_action(image, selection, action_type='fling-to-fold', save=True)
        pixel1 = PlanarTransform.fromRelativePixelDictionary(action['poses'][0], image.shape)
        pixel2 = PlanarTransform.fromRelativePixelDictionary(action['poses'][1], image.shape)
        points_3d = exp.get_pointcloud(image)
        pick1 = exp.pixel_to_transform(pixel1, image.shape, points_3d)
        pick2 = exp.pixel_to_transform(pixel2, image.shape, points_3d)

        transform_bottom = exp.get_transform(x=-0.20, frame='world')
        transform_sleeve = exp.get_transform(x=-0.28, frame='world')

        if pick1.translation[0] < pick2.translation[0]:
            place1 = transform_bottom * pick1
            place2 = transform_sleeve * pick2
        else:
            place1 = transform_sleeve * pick1
            place2 = transform_bottom * pick2

        exp.pick_and_place(pick1, pick2, place1, place2)
        exp.home()

    if args.do_2s_fold:
        image, _ = exp.take_image()
        exp.execute_2s_fold(image)
        exp.home()

    if args.do_final_fling:
        pick1 = Experiment.get_transform(x=0.36, y=0.19, z=0.055, a=np.pi, b=0.0, c=-np.pi, from_frame='l_tcp')
        pick2 = Experiment.get_transform(x=0.35, y=-0.06, z=0.055, a=np.pi, b=0.0, c=np.pi, from_frame='r_tcp')

        exp.final_fling(pick1, pick2)
        exp.home()

    if args.do_pick_and_hold_primitive:
        pick1 = Experiment.get_transform(x=0.36, y=-0.05, z=0.048, a=np.pi, b=0.0, c=np.pi, from_frame='robot')
        hold2 = Experiment.get_transform(x=0.32, y=-0.12, z=0.045, a=np.pi, b=0.0, c=-np.pi, from_frame='robot')
        place1 = Experiment.get_transform(x=0.44, y=0.20, z=0.048, a=np.pi, b=0.0, c=np.pi, from_frame='robot')

        exp.pick_and_hold(pick1, place1, hold2)
        exp.home()

    if args.do_move_primitive:
        exp.move_to_center()
        exp.home()

    if args.instruction:
        image_normal, image_ortho = exp.take_image()
        mask, _ = segment(image_normal.color.data)
        instruction = exp.tm.get_matched_instruction(mask, template_name=args.instruction)
        exp.execute_fold(instruction, image_normal)

    if args.check_calibration:
        pixel = (686, 264)

        image_normal, _ = exp.take_image()
        points_3d = exp.get_pointcloud(image_normal)
        pose = exp.pixel_to_transform(pixel, image_normal.shape, points_3d)
        exp.y.left.goto_pose(pose, speed=exp.half_speed)
        exp.y.left.sync()

    if args.check_reachability:
        pose_left = Experiment.get_transform(x=0.25, y=-0.16, z=0.06, a=np.pi, b=0.0, c=np.pi + 0.2, from_frame='l_tcp')
        pose_right = Experiment.get_transform(x=0.45, y=-0.15, z=0.06, a=np.pi, b=0.0, c=np.pi - 0.9, from_frame='r_tcp')

        pose_left, pose_right, _, _, _, _ = exp.optimize_angle_a_for_reachability(pose_left, pose_right)
        
        exp.y.left.goto_pose(pose_left, speed=exp.full_speed)
        exp.y.right.goto_pose(pose_right, speed=exp.full_speed)

        exp.y.left.sync()
        exp.y.right.sync()

    if args.exit:
        with open(exp.local_data_path / 'exits' / args.exit / 'information.txt', 'r') as f:
            action_type = f.readline().strip()
        
        if action_type == 'fling':
            pick1 = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-0.tf')
            pick2 = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-1.tf')
            exp.fling(pick1, pick2)

        elif action_type == 'pick-and-hold':
            pick = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-0.tf')
            place = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-1.tf')
            hold = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-2.tf') if (exp.local_data_path / 'exits' / args.exit / 'pose-2.tf').exists() else None
            exp.pick_and_hold(pick, place, hold)

        elif action_type == 'pick-and-place':
            pick1 = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-0.tf')
            pick2 = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-1.tf')
            place1 = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-2.tf')
            place2 = RigidTransform.load(exp.local_data_path / 'exits' / args.exit / 'pose-3.tf')
            exp.pick_and_place(pick1, pick2, place1, place2)

        else:
            raise Exception('action type not implemented yet')
        
        exp.home()

    if args.gen_rand_scene:
        image_normal, _ = exp.take_image()
        exp.gen_random_scene(image_normal)
        exp.home()
