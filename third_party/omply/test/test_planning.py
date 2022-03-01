import time

import numpy as np

from autolab_core import RigidTransform
from autolab_core import transformations as tr
from omply import RobotModel


def get_transform(x=0.0, y=0.0, z=0.0, a=0.0, b=0.0, c=0.0, to_frame='world', from_frame='unassigned', frame=None):
    to_frame = frame if frame else to_frame
    from_frame = frame if frame else from_frame
    return RigidTransform(translation=[x, y, z], rotation=tr.euler_matrix(a, b, c)[:3,:3], to_frame=to_frame, from_frame=from_frame)


ABB_WHITE = RigidTransform(translation=[0, 0, 0.1325])
frame_left = ABB_WHITE.as_frames(RobotModel.l_tcp_frame, RobotModel.l_tip_frame)
frame_right = ABB_WHITE.as_frames(RobotModel.r_tcp_frame, RobotModel.r_tip_frame)


model = RobotModel()
model.set_tcp(frame_left, frame_right)

s = time.time()

pick1 = get_transform(x=0.56, y=0.15, z=0.055, a=np.pi, b=0.0, c=-np.pi, from_frame='l_tcp')
pick2 = get_transform(x=0.57, y=-0.15, z=0.055, a=np.pi, b=0.0, c=np.pi, from_frame='r_tcp')
joints = model.ik(pick1, pick2)
is_valid = model.is_valid_state(joints[0], joints[1])
is_valid = model.get_distance(joints[0], joints[1])
print(is_valid, joints)

e = time.time()

print(f'Computational time: {(e - s)*1000:0.5f} [ms]')
