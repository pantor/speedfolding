import json
import multiprocessing as mp
import numpy as np
import queue as Queue
import time

import abb_librws as abb
from autolab_core import RigidTransform


MM_TO_M = 1.0 / 1000.0
M_TO_MM = 1000.0

SPEEDDATA_CONSTANTS = [
    "v5", "v10", "v20", "v30", "v40", "v50", "v60", "v80",
    "v100", "v150", "v200", "v300", "v400", "v500", "v600", "v800",
    "v1000", "v1500", "v2000", "v2500", "v3000", "v4000", "v5000", "v6000", "v7000", "vmax",
]

ZONEDATA_CONSTANTS = [
    "fine", "z0", "z1", "z5", "z10", "z15", "z20", "z30",
    "z40", "z50", "z60", "z80", "z100", "z150", "z200",
]


#TODO add sync option for all motions
#TODO exception handling of motion supervision
SLEEP_TIME = 0.05


def cmd(func, args, tries=10):
    for _ in range(tries):
        try:
            return func(*args)
        except RuntimeError:
            print(f"yumi.py: retrying cmd {func}")
            time.sleep(.03)
    raise RuntimeError(f"Couldn't execute command {func}")


class YuMi:
    def __init__(self, l_tcp=RigidTransform(), r_tcp=RigidTransform(), ip_address="192.168.125.1"):
        try:
            self._iface = abb.RWSStateMachineInterface(ip_address)
        except RuntimeError:
            print("YuMi could not connect!")
        
        r_task, l_task = self._iface.rapid_tasks

        self._lock = mp.Lock()
        cmd(self.stop_rapid,())
        cmd(self.reset_program_pointer,())
        cmd(self.start_rapid,())

        self.left = YuMiArm(self._lock, ip_address, l_task.name, l_tcp)
        self.left.daemon = True
        self.left.start()

        self.right = YuMiArm(self._lock, ip_address, r_task.name, r_tcp)
        self.right.daemon = True
        self.right.start()

    @property
    def auto_mode(self):
        with self._lock:
            return self._iface.auto_mode

    @property
    def connected(self):
        with self._lock:
            return self._iface.runtime_info.rws_connected

    @property
    def motors_on(self):
        with self._lock:
            return self._iface.motors_on

    @motors_on.setter
    def motors_on(self, value):
        with self._lock:
            self._iface.set_motors_on() if value else self._iface.set_motors_off()

    @property
    def rapid_running(self):
        with self._lock:
            return self._iface.rapid_running

    @property
    def rw_version(self):
        with self._lock:
            return self._iface.system_info.robot_ware_version

    @property
    def speed_ratio(self):
        with self._lock:
            return self._iface.get_speed_ratio()

    @speed_ratio.setter
    def speed_ratio(self, value):
        with self._lock:
            self._iface.set_speed_ratio(value)

    @property
    def system_name(self):
        with self._lock:
            return self._iface.system_info.system_name

    @property
    def system_options(self):
        with self._lock:
            return self._iface.system_info.system_options

    def log_text(self, verbose=False):
        with self._lock:
            return self._iface.log_text(verbose)

    def log_text_latest(self):
        with self._lock:
            return self._iface.log_text_latest()

    def start_rapid(self):
        if not self.motors_on:
            self.motors_on = True
            time.sleep(.3)
        
        with self._lock:
            self._iface.start_rapid()
            time.sleep(0.1)

    def stop_rapid(self):
        with self._lock:
            self._iface.stop_rapid()
            time.sleep(.1)

    def reset_program_pointer(self):
        with self._lock:
            self._iface.reset_program_pointer()
            time.sleep(.1)

    def calibrate_grippers(self):
        self._gripper_fn("calibrate")
        time.sleep(5)

    def move_grippers(self, lpos, rpos):
        self._gripper_fn("move_to", lpos * M_TO_MM, rpos * M_TO_MM)

    def close_grippers(self):
        self._gripper_fn("grip_in")

    def open_grippers(self):
        self._gripper_fn("grip_out")

    def _gripper_fn(self, fn_name, *args):
        with self._lock:
            return getattr(self._iface.services().sg(), f"dual_{fn_name}")(*args)

    def move_joints_sync(self, l_joints, r_joints, speed=(0.3, 2 * np.pi), zone="z1", final_zone="fine", minimum_height=None):
        assert len(l_joints) == len(r_joints), "Sync move must have equal joint traj lengths"

        self.left.move_joints_sync(l_joints, speed, zone, final_zone, minimum_height)
        self.right.move_joints_sync(r_joints, speed, zone, final_zone, minimum_height)

    def move_cartesian_sync(self, left_poses, right_poses, speed=(0.3, 2 * np.pi), zone="z0", final_zone="fine"):
        assert len(left_poses) == len(right_poses), "Sync move must have equal joint traj lengths"

        self.left.move_cartesian_sync(left_poses, speed, zone, final_zone)
        self.right.move_cartesian_sync(right_poses, speed, zone, final_zone)


class YuMiArm(mp.Process):
    class Decorators:
        @classmethod
        def add_to_queue(cls, func):
            def wrapper(self, *args, **kwargs):
                with self._q_len.get_lock():
                    self._q_len.value += 1
                self._input_queue.put((func.__name__, args, kwargs))
            wrapper.__wrapped__ = func
            return wrapper

    def __init__(self, lock, ip_address, task, tcp=RigidTransform()):
        super().__init__()
        self._input_queue = mp.Queue()
        self._q_len = mp.Value("i", 0)  # we need this to be a reliable counter for the q size
        self._iface = abb.RWSStateMachineInterface(ip_address)
        self._task = task
        self._tcp = tcp

        self._side = "left" if self._task.lower()[-1] == "l" else "right"
        self._custom_mod = abb.FileResource(f"custom_{self._task.lower()}.sys")
        self._custom_mod_path = f"HOME:/{self._custom_mod.filename}"
        self._lock = lock
        tooltip = f'''
        MODULE {self._task}_tcp
            TASK PERS tooldata tool{self._task.lower()} := {self.tool_str};
            PERS tasks task_list{{2}} := [["T_ROB_L"],["T_ROB_R"]];
            TASK PERS bool pending_move_err := FALSE;
            TASK PERS errnum lasterr := 42;
        ENDMODULE'''

        tooltipmod = abb.FileResource(f"tooltip_{self._task.lower()}.sys")
        tooltippath = f"HOME:/{tooltipmod.filename}"
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.services().rapid().run_module_unload(self._task,tooltippath)
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.upload_file(tooltipmod, tooltip)
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.services().rapid().run_module_load(self._task, tooltippath)
        time.sleep(SLEEP_TIME)

    def err_handler(self):
        return f'''
        ERROR
            lasterr := ERRNO;
            pending_move_err := TRUE;
            StopMoveReset;'''
    
    def run(self):
        while True:
            try:
                func, args, kwargs = self._input_queue.get(timeout=1)
            except Queue.Empty:
                continue

            getattr(self, func).__wrapped__(self, *args, **kwargs)
            with self._q_len.get_lock():
                self._q_len.value -= 1

    def sync(self, timeout=float('inf')):
        start = time.time()
        while self._q_len.value > 0 and (time.time() - start < timeout):
            pass
        return time.time() - start > timeout

    @property
    def tcp(self):
        return self._tcp

    @tcp.setter
    def tcp(self, value):
        self._tcp = value

    @property
    def tool_str(self):
        """returns the tooltip string for the current TCP"""
        t = (self._tcp.translation * M_TO_MM).astype(str)
        q = self._tcp.quaternion.astype(str)
        return f"[TRUE,[[{','.join(t)}],[{','.join(q)}]],[0.001,[0,0,0.001],[1,0,0,0],0,0,0]]"

    def get_joints(self):
        try:
            with self._lock:
                jt = self._iface.mechanical_unit_joint_target(self._task[2:])
        except RuntimeError:
            return self.get_joints()
        
        return np.deg2rad([
            jt.robax.rax_1.value,
            jt.robax.rax_2.value,
            jt.extax.eax_a.value,
            jt.robax.rax_3.value,
            jt.robax.rax_4.value,
            jt.robax.rax_5.value,
            jt.robax.rax_6.value,
        ])
        
    def clear_error(self):
        with self._lock:
            err = cmd(self._iface.get_rapid_symbol_data, (self._task, f"{self._task.lower()}_tcp", "pending_move_err"))
            errnum = cmd(self._iface.get_rapid_symbol_data, (self._task, f"{self._task.lower()}_tcp", "lasterr"))
            if err == 'TRUE':
                cmd(self._iface.set_rapid_symbol_data, (self._task, f"{self._task.lower()}_tcp", "pending_move_err", "FALSE"))
        return err == 'TRUE', errnum

    def calibrate_gripper(self):
        self.gripper_fn("calibrate")

    def initialize_gripper(self):
        self.gripper_fn("initialize")

    def open_gripper(self):
        self.gripper_fn("grip_out")

    def close_gripper(self):
        self.gripper_fn("grip_in")

    def move_gripper(self, value):
        self.gripper_fn("move_to", M_TO_MM * value)

    @property
    def gripper_settings(self):
        return self.gripper_fn.__wrapped__(self, "get_settings")

    @gripper_settings.setter
    def gripper_settings(self, value):
        self.gripper_fn.__wrapped__(self, "set_settings", value)

    def get_ik(self, poses):
        varstr = ''
        calcstr = ''

        for i, p in enumerate(poses):
            varstr += f"""
                VAR jointtarget outjt{i};
                VAR errnum myerrnum{i};
                """
            calcstr += f"""
                outjt{i} := CalcJointT({self._get_rapid_pose(p)}, tool{self._task.lower()}, \ErrorNumber:=myerrnum{i});"""

        calcstr += f"""
            ERROR
                IF ERRNO = ERR_ROBLIMIT THEN
                    SkipWarn;
                    TRYNEXT;
                ENDIF"""

        self._execute_custom(varstr, calcstr)

        result = []
        for i, _ in enumerate(poses):
            try:
                ret = json.loads(self._iface.get_rapid_symbol_data(self._task, 'customModule', f'outjt{i}'))
                result.append(np.deg2rad(ret[0]))
            except RuntimeError:
                result.append(None)
        return result

    def _get_rapid_speeds(self, speed, number_waypoints=1):
        result = None
        if isinstance(speed, str):
            result = [speed] * number_waypoints
        elif isinstance(speed, (np.ndarray, list, tuple)):
            if isinstance(speed[0], (np.ndarray, list, tuple)):
                result = [abb.SpeedData((s[0] * M_TO_MM, np.rad2deg(s[1]), 5000, 5000)) for s in speed]
            elif isinstance(speed[0], str):
                result = speed
            else:
                result = [abb.SpeedData((speed[0] * M_TO_MM, np.rad2deg(speed[1]), 5000, 5000))] * number_waypoints

        assert result is not None, 'Speed must either be a single string or a (2,) or (n,2) iterable'
        return result

    def _get_rapid_zones(self, zone, number_waypoints=1, final_zone=None):
        result = None
        if isinstance(zone, str):
            result = [zone] * number_waypoints
        elif isinstance(zone, (np.ndarray, list, tuple)):
            if isinstance(zone[0], (np.ndarray, list, tuple)):
                result = [abb.ZoneData(z) for z in zone]
            elif isinstance(zone[0], str):
                result = zone
            else:
                result = [abb.ZoneData(zone)] * number_waypoints

        if final_zone:
            result[-1] = final_zone if isinstance(final_zone, str) else abb.ZoneData(final_zone)
        
        assert result is not None, 'Zone must either be a single string or a (7,) or (n,7) iterable'
        return result
    
    def _get_rapid_pose(self, pose, cf6=0):
        with self._lock:
            time.sleep(SLEEP_TIME)
            rt = self._iface.mechanical_unit_rob_target(self._task[2:], abb.Coordinate.BASE, "tool0", "wobj0")
        
        trans = pose.translation * M_TO_MM
        rt.pos = abb.Pos(trans)
        rt.orient = abb.Orient(*pose.quaternion)
        rt.robconf.cf6 = abb.RAPIDNum(cf6)
        return rt

    @Decorators.add_to_queue
    def move_joints_sync(self, joints, speed, zone, final_zone, minimum_height):
        joints = np.rad2deg(joints)
        speeds = self._get_rapid_speeds(speed, len(joints))
        zones = self._get_rapid_zones(zone, len(joints), final_zone=final_zone)

        toolstr = f'''
        VAR robtarget current;
        VAR syncident sync1;
        VAR syncident sync2;
        '''

        wpstr = ''
        # Moves to a given minimum height within the workspace with a linear motion first
        if minimum_height:
            wpstr += f'''
            current := CRobT(\Tool:=tool{self._task.lower()});
            IF current.trans.z < 60 THEN
                current.trans.z := current.trans.z + 15;
                MoveL current, {speeds[0]}, {zones[0]}, tool{self._task.lower()};
            ENDIF
            IF current.trans.y {'< -60' if self._side == 'left' else '> 60'} THEN
                current.trans.y := {'-60' if self._side == 'left' else '60'};
                MoveL current, {speeds[0]}, {zones[0]}, tool{self._task.lower()};
            ENDIF
            IF current.trans.x < 220 AND current.trans.z < 400 THEN
                current.trans.x := 220;
                MoveL current, {speeds[0]}, {zones[0]}, tool{self._task.lower()};
            ENDIF
            IF current.trans.x < 280 AND current.trans.z < 400 AND current.trans.y {'< 0' if self._side == 'left' else '> 0'} THEN
                current.trans.y := {'0' if self._side == 'left' else '0'};
                current.trans.x := 280;
                MoveL current, {speeds[0]}, {zones[0]}, tool{self._task.lower()};
            ENDIF
            IF current.trans.z + 15 < {minimum_height * M_TO_MM} AND current.trans.x > 219 THEN
                current.trans.z := {minimum_height * M_TO_MM};
                MoveL current, {speeds[0]}, {zones[0]}, tool{self._task.lower()};
            ENDIF
            '''

        wpstr += '''
        SyncMoveOn sync1, task_list;'''

        for i, (wp, sd, zd) in enumerate(zip(joints, speeds, zones)):
            jt = abb.JointTarget(abb.RobJoint(np.append(wp[:2], wp[3:])), abb.ExtJoint(eax_a=wp[2]))
            wpstr += f"\t\tMoveAbsJ {jt}, \ID:={i}, {sd}, {zd}, tool{self._task.lower()};\n"
        
        wpstr += f"\t\tSyncMoveOff sync2;"
        self._execute_custom(toolstr, wpstr)

    @Decorators.add_to_queue
    def move_joints_traj(self, joints, speed=(0.3, 2 * np.pi), zone="z1", final_zone="fine"):
        joints = np.rad2deg(joints)
        speeds = self._get_rapid_speeds(speed, len(joints))
        zones = self._get_rapid_zones(zone, len(joints), final_zone=final_zone)

        # Create RAPID code and execute
        wpstr = ''
        for wp, sd, zd in zip(joints, speeds, zones):
            jt = abb.JointTarget(abb.RobJoint(np.append(wp[:2], wp[3:])), abb.ExtJoint(eax_a=wp[2]))
            wpstr += f"\t\tMoveAbsJ {jt}, {sd}, {zd}, tool{self._task.lower()};\n"
        
        self._execute_custom('', wpstr)

    @Decorators.add_to_queue
    def goto_pose(self, pose, speed=(0.3, 2 * np.pi), zone="fine", linear=True, cf6=0):
        sd = self._get_rapid_speeds(speed)[0]

        toolstr = f"\n\tVAR robtarget p1 := {self._get_rapid_pose(pose, cf6=cf6)};"
        cmd = "MoveL" if linear else "MoveJ"
        wpstr = f"\t\t{cmd} p1, {sd}, {zone}, tool{self._task.lower()};"
        self._execute_custom(toolstr, wpstr)

    @Decorators.add_to_queue
    def move_cartesian_sync(self, poses, speed, zone, final_zone):
        speeds = self._get_rapid_speeds(speed, len(poses))
        zones = self._get_rapid_zones(zone, len(poses), final_zone=final_zone)

        toolstr = ''
        for i, p in enumerate(poses):
            toolstr += f'\n\tVAR robtarget p{i} := {self._get_rapid_pose(p)};'
        
        wpstr = '''
        VAR syncident sync1;
        VAR syncident sync2;
        SyncMoveOn sync1, task_list;'''

        for i, (p, sd, zd) in enumerate(zip(poses, speeds, zones)):
            wpstr += f'\n\t\tMoveL p{i}, \ID:={i}, {sd}, {zd}, tool{self._task.lower()};'

        wpstr += f"\n\t\tSyncMoveOff sync2;"
        self._execute_custom(toolstr, wpstr)

    @Decorators.add_to_queue
    def move_cartesian_traj(self, poses, speed=(0.3, 2 * np.pi), zone='z0', final_zone='fine'):
        speeds = self._get_rapid_speeds(speed, len(poses))
        zones = self._get_rapid_zones(zone, len(poses), final_zone=final_zone)

        toolstr = ''
        for i, p in enumerate(poses):
            toolstr += f'\n\tVAR robtarget p{i} := {self._get_rapid_pose(p)};'

        wpstr = ''
        for i, (p, sd, zd) in enumerate(zip(poses, speeds, zones)):
            wpstr += f'\n\t\tMoveL p{i}, {sd}, {zd}, tool{self._task.lower()};'
        
        self._execute_custom(toolstr, wpstr)

    @Decorators.add_to_queue
    def move_contact(self, pose, speed=(0.3, 2 * np.pi), max_torque=1.0, reaction=8): # [mm]
        sd = self._get_rapid_speeds(speed)[0]

        toolstr = f'''
        VAR num desiredTorque;
        VAR bool collisionDetected;
        VAR robtarget p0 := {self._get_rapid_pose(pose)};
        VAR robtarget pnew;
        VAR pos diff;'''

        wpstr = f'''
        desiredTorque := {max_torque};
        collisionDetected := TRUE;
        ContactL \DesiredTorque := desiredTorque, p0, {sd}, tool{self._task.lower()};
        IF collisionDetected THEN
            pnew := CRobT(\Tool:=tool{self._task.lower()});
            diff := p0.trans - pnew.trans;
            pnew.trans.x := pnew.trans.x - {reaction} * diff.x / Distance(p0.trans, pnew.trans);
            pnew.trans.y := pnew.trans.y - {reaction} * diff.y / Distance(p0.trans, pnew.trans);
            pnew.trans.z := pnew.trans.z - {reaction} * diff.z / Distance(p0.trans, pnew.trans);
            ContactL pnew, {sd}, tool{self._task.lower()};
        ENDIF
        ERROR
            IF ERRNO=ERR_CONTACTL THEN
                SkipWarn;
                collisionDetected := FALSE;
                TRYNEXT;
            ENDIF
        '''
        self._execute_custom(toolstr, wpstr)

    def get_pose(self):
        with self._lock:
            time.sleep(SLEEP_TIME)
            rt = cmd(self._iface.mechanical_unit_rob_target,(self._task[2:], abb.Coordinate.BASE, "tool0", "wobj0"))
        
        trans = MM_TO_M * np.array([rt.pos.x.value, rt.pos.y.value, rt.pos.z.value])
        q = np.array([rt.orient.q1.value, rt.orient.q2.value, rt.orient.q3.value, rt.orient.q4.value])
        wrist = RigidTransform(
            translation=trans,
            rotation=RigidTransform.rotation_from_quaternion(q),
            from_frame=self._tcp.to_frame,
            to_frame="base_link",
        )
        return wrist * self._tcp

    @Decorators.add_to_queue
    def gripper_fn(self, fn_name, *args):
        with self._lock:
            res = getattr(self._iface.services().sg(), f"{self._side}_{fn_name}")(*args)
            time.sleep(SLEEP_TIME)
            return res

    def _execute_custom(self, toolstr: str, wpstr: str):
        routine = f'''
        MODULE customModule{toolstr}
            PROC custom_routine0()
            {wpstr}
            ENDPROC
        ENDMODULE'''

        # Upload and execute custom routine (unloading needed for new routine)
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.services().rapid().run_module_unload(self._task, self._custom_mod_path)
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.upload_file(self._custom_mod, routine)
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.services().rapid().run_module_load(self._task, self._custom_mod_path)
        time.sleep(SLEEP_TIME)
        with self._lock:
            self._wait_for_cmd()
            self._iface.services().rapid().run_call_by_var(self._task, 'custom_routine', 0)
        time.sleep(SLEEP_TIME)
        self._wait_for_cmd_lock()

    def _wait_for_cmd(self):
        while True:
            bad = not cmd(self._iface.services().main().is_idle, (self._task,)) or not \
                cmd(self._iface.services().main().is_stationary, (self._task[2:],))
            if not bad: break

    def _wait_for_cmd_lock(self):
        while True:
            with self._lock:
                bad = not cmd(self._iface.services().main().is_idle,(self._task,)) or not \
                    cmd(self._iface.services().main().is_stationary,(self._task[2:],))
            if not bad: break
            time.sleep(SLEEP_TIME)
