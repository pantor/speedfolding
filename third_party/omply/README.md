# Omply 

This is a library for trajectory planning and kinematics, designed for the YuMi robot


## Installation

It depends on the Tracikpy inverse kinematics library: https://github.com/mjd3/tracikpy. Follow the instructions there to install

To install, clone and call `pip install -e .` on this directory.


## Usage

```.py
from omply import RobotModel

model = RobotModel()

joints_left, joints_right = model.ik(pose_left, pose_right)
is_valid = model.is_valid_state(joints_left, joints_right)
distance = model.get_distance(joints_left, joints_right)

```