import os

from autolab_core import RigidTransform
import numpy as np

from ._omply import CollisionChecker, DualArmPlanner, SingleArmPlanner
from ._omply import TracIK, SolveType


class RobotModel:
    base_frame = 'base_link'
    # tip frame is the end of the urdf file, in this case meaning the wrist
    l_tip_frame, r_tip_frame = 'gripper_l_base', 'gripper_r_base'
    # tcp is tool center point, meaning the point ik and fk will compute to
    l_tcp_frame, r_tcp_frame = 'l_tcp', 'r_tcp'

    LEFT_BASE = RigidTransform(translation=[0.05, 0.64, 0.4])
    RIGHT_BASE = RigidTransform(translation=[0.05, -0.64, 0.4])

    # Only for flingbot ablative study as poses are usually nearer then!
    # LEFT_BASE = RigidTransform(translation=[0.05, 10.94, 0.4])
    # RIGHT_BASE = RigidTransform(translation=[0.05, -10.94, 0.4])

    L_HOME_STATE = np.array([-0.5810662, -1.34913424, 0.73567095, 0.55716616, 1.56402364, 1.25265177, 0.0])
    R_HOME_STATE = np.array([0.64224786, -1.34920282, -0.82859683, 0.52531042, -1.64836569, 1.20916355, 0.0])

    def __init__(self, robot_name='yumi', timeout=0.008, solve_type=SolveType.Speed):
        '''
        solve type (taken from tracik):
        - Speed: returns very quickly the first solution found
        - Distance: runs for the full timeout_in_secs, then returns the solution that minimizes SSE from the seed
        - Manipulation1: runs for full timeout, returns solution that maximizes sqrt(det(J*J^T)) (the product of the singular values of the Jacobian)
        - Manipulation2: runs for full timeout, returns solution that minimizes the ratio of min to max singular values of the Jacobian.
        
        '''

        desc_path = os.path.dirname(os.path.abspath(__file__)) + f'/../descriptions/{robot_name}/'
        urdf_path = os.path.dirname(os.path.abspath(__file__)) + f'/../descriptions/{robot_name}/urdf/{robot_name}.urdf'
        
        self.coll = CollisionChecker(desc_path)
        self.dual = DualArmPlanner(self.coll)
        self.left = SingleArmPlanner(self.coll, True, True)
        self.right = SingleArmPlanner(self.coll, False, True)

        #setup the tool center point as 0 transform
        self.set_tcp()

        _urdf_string = "".join(open(urdf_path, "r").readlines())
        self.left_solver = TracIK(self.base_frame, self.l_tip_frame, _urdf_string, timeout=timeout, solve_type=solve_type)
        self.right_solver = TracIK(self.base_frame, self.r_tip_frame, _urdf_string, timeout=timeout, solve_type=solve_type)
        
        self.left_solver.joint_limits = self.left_solver.joint_limits[0] + 1e-5, self.left_solver.joint_limits[1] - 1e-5
        self.right_solver.joint_limits = self.right_solver.joint_limits[0] + 1e-5, self.right_solver.joint_limits[1] - 1e-5
        
        self.left_joint_lims = self.left_solver.joint_limits
        self.right_joint_lims = self.right_solver.joint_limits

    def plan(self, l_start, r_start, l_goal=None, r_goal=None, timeout=1):
        '''
        plan a path from starts to goals. if only one goal is specified, planning will be done
        with single arm planner (other arm doesn't move). if two goals are specified, both arms
        will be planned for at once
        '''
        if l_goal is not None and r_goal is not None:
            #plan both
            s = np.zeros(14)
            s[:7] = l_start;s[7:] = r_start
            g = np.zeros(14)
            g[:7] = l_goal;g[7:] = r_goal
            path = self.dual.planPath(s,g,timeout)
            if len(path) == 0:
                raise Exception("Couldn't plan path")
            path = np.array(path)
            return path[:,:7], path[:,7:]

        if l_goal is not None:
            path = self.left.planPath(l_start,l_goal,r_start,timeout)
            if len(path) == 0:
                raise Exception("Couldn't plan path")
            return np.array(path), []

        if r_goal is not None:
            path = self.right.planPath(r_start,r_goal,l_start,timeout)
            if len(path) == 0:
                raise Exception("Couldn't plan path")
            return [], np.array(path)

    def plan_to_pose(self, l_start_q, r_start_q, l_goal_p=None, r_goal_p=None):
        '''Returns path(s) from the start location to goal poses'''
        l_goal, r_goal = self.find_joints(l_start_q, r_start_q, 200, l_goal_p, r_goal_p)
        if l_goal is None and r_goal is None:
            raise Exception("Couldn't find valid goal states to reach goal poses")
        return self.plan(l_start_q, r_start_q, l_goal, r_goal)

    def find_joints(self, l_start_q, r_start_q, samples, l_goal_p=None, r_goal_p=None):
        def isvalidiksol(lsol, rsol):
            if l_goal_p is None:
                lsol = l_start_q
            if r_goal_p is None:
                rsol = r_start_q
            if lsol is None or rsol is None:
                return False
            return not self.coll.isColliding(lsol, rsol)
        
        if l_start_q is not None or r_start_q is not None:
            l_goal, r_goal = self.ik(l_goal_p, r_goal_p, l_start_q, r_start_q)
            unseeded = False
        else:
            unseeded = True

        fixedsamples = [(np.zeros(7), np.zeros(7))]
        if unseeded or not isvalidiksol(l_goal, r_goal):
            for i in range(samples):
                #try randomly a few times to see if we find one that doesn't collide
                if i < len(fixedsamples):
                    li, ri = fixedsamples[i]
                else:
                    li = np.random.uniform(self.left_joint_lims[0], self.left_joint_lims[1])
                    ri = np.random.uniform(self.right_joint_lims[0], self.right_joint_lims[1])
                l_goal, r_goal = self.ik(l_goal_p, r_goal_p, li, ri)
                if isvalidiksol(l_goal, r_goal):
                    break
        if not isvalidiksol(l_goal, r_goal):
            return None, None
        return l_goal, r_goal

    def is_valid_state(self, l_q, r_q):
        return l_q is not None and r_q is not None and self.coll.isInBounds(l_q, r_q) and not self.coll.isColliding(l_q,r_q)

    def get_distance(self, left_joints, right_joints):
        if left_joints is None or right_joints is None:
            return 0.0

        return self.coll.getDistance(left_joints, right_joints)

    def set_tcp(self, l_tool=None, r_tool=None):
        if l_tool is None:
            self.l_tcp = RigidTransform(from_frame=self.l_tcp_frame, to_frame=self.l_tip_frame)
        else:
            assert l_tool.from_frame == self.l_tcp_frame, l_tool.to_frame == self.l_tip_frame
            self.l_tcp = l_tool
        
        if r_tool is None:
            self.r_tcp = RigidTransform(from_frame=self.r_tcp_frame, to_frame=self.r_tip_frame)
        else:
            assert r_tool.from_frame == self.r_tcp_frame, r_tool.to_frame == self.r_tip_frame
            self.r_tcp = r_tool

    @classmethod
    def ik_single(cls, solver, pose: RigidTransform, tcp, qinit, bs):
        if pose is None:
            return None
        
        pose = pose * tcp.inverse()
        jmin, jmax = solver.joint_limits
        qinit = np.clip(qinit, jmin, jmax)
        return solver.ik(pose.matrix, qinit, bs)

    def ik(self, left_pose, right_pose, left_qinit=None, right_qinit=None, bs=[1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3]):
        '''
        given left and/or right target poses, calculates the joint angles and returns them as a tuple
        poses are RigidTransforms, qinits are np arrays
        solve_type can be "Distance" or "Manipulation1" or "Speed" (See constructor for what these mean)
        bs is an array representing the tolerance on end pose of the gripper
        '''

        left_qinit = left_qinit if left_qinit is not None else self.L_HOME_STATE
        right_qinit = right_qinit if right_qinit is not None else self.R_HOME_STATE

        left_result = self.ik_single(self.left_solver, left_pose, self.l_tcp, left_qinit, bs)
        right_result = self.ik_single(self.right_solver, right_pose, self.r_tcp, right_qinit, bs)
        return left_result, right_result

    def fk(self, qleft=None, qright=None):
        '''
        computes the forward kinematics for left and right arms. 
        qleft,qright are np arrays 
        returns a tuple of RigidTransform
        '''
        lres, rres = None, None

        if qleft is not None:
            lpos = self.left_solver.fk(qleft)
            lres = RigidTransform(translation=lpos[:3,3], rotation=lpos[:3,:3], from_frame=self.l_tip_frame, to_frame=self.base_frame) * self.l_tcp
        
        if qright is not None:
            rpos = self.right_solver.fk(qright)
            rres = RigidTransform(translation=rpos[:3,3], rotation=rpos[:3,:3], from_frame=self.r_tip_frame, to_frame=self.base_frame) * self.r_tcp
        return lres, rres
