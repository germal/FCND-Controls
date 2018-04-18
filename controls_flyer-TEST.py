# -*- coding: utf-8 -*-
"""
Starter code for the controls project.
This is the solution of the backyard flyer script, 
modified for all the changes required to get it working for controls.
"""

import time
from enum import Enum

import numpy as np

from udacidrone import Drone
from unity_drone import UnityDrone
from controller import NonlinearController
from udacidrone.connection import MavlinkConnection  # noqa: F401
from udacidrone.messaging import MsgID

TRAJECTORY = 0

class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class ControlsFlyer(UnityDrone):

    def __init__(self, connection):
        super().__init__(connection)
        self.controller = NonlinearController()
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL
        self.prev_time = time.time()
        self.att_time = 0.0
        self.pos_time = 0.0
        self.pulse = False

        # register callbacks
        self.register_callback(MsgID.LOCAL_POSITION,
                               self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)
        
        self.register_callback(MsgID.ATTITUDE, self.attitude_callback)
        self.register_callback(MsgID.RAW_GYROSCOPE, self.gyro_callback)
        
    def position_controller(self):  
        (self.local_position_target,
         self.local_velocity_target,
         yaw_cmd, accff) = self.controller.trajectory_control(
                 self.position_trajectory,
                 self.yaw_trajectory,
                 self.time_trajectory, time.time(), True)
        self.attitude_target = np.array((0.0, 0.0, yaw_cmd))
        cur_time = time.time()
        acceleration_cmd = self.controller.lateral_position_control(
                self.local_position_target[0:2],
                self.local_velocity_target[0:2],
                self.local_position[0:2],
                self.local_velocity[0:2],
                accff[0:2], 
                cur_time-self.prev_time)
        self.prev_time = cur_time
        self.local_acceleration_target = np.array([acceleration_cmd[0],
                                                   acceleration_cmd[1],
                                                   0.0])
        
    def attitude_controller(self):
        self.thrust_cmd = self.controller.altitude_control(
                -self.local_position_target[2],
                -self.local_velocity_target[2],
                -self.local_position[2],
                -self.local_velocity[2],
                self.attitude,
                9.81)
        roll_pitch_rate_cmd = self.controller.roll_pitch_controller(
                self.local_acceleration_target[0:2],
                self.attitude,
                self.thrust_cmd)
        if self.pulse:
            roll_pitch_rate_cmd[0] = 30.0
            roll_pitch_rate_cmd[1] = 0.0
            self.pulse = False
        yawrate_cmd = self.controller.yaw_control(
                self.attitude_target[2],
                self.attitude[2])
        self.body_rate_target = np.array(
                [roll_pitch_rate_cmd[0], roll_pitch_rate_cmd[1], yawrate_cmd])
        
    def bodyrate_controller(self):        
        moment_cmd = self.controller.body_rate_control(
                self.body_rate_target,
                self.gyro_raw)
        self.cmd_moment(moment_cmd[0],
                        moment_cmd[1],
                        moment_cmd[2],
                        self.thrust_cmd)
    
    def attitude_callback(self): # 40 frames per second
        if time.time()-self.att_time > 0.05: # pace the calls to 20fps
            if self.flight_state == States.WAYPOINT:
                self.attitude_controller()
                self.att_time = time.time()
    
    def gyro_callback(self): # 40 frames per second
        if self.flight_state == States.WAYPOINT:
            self.bodyrate_controller()

    def local_position_callback(self): # 40 frames per second
        #
        #print(time.time())
        if self.flight_state == States.TAKEOFF:
            if abs(self.local_position[2]+self.target_position[2])<0.025 and np.linalg.norm(self.local_velocity[0:2])<0.1:
                if TRAJECTORY is 1:
                    (self.position_trajectory, self.time_trajectory,
                     self.yaw_trajectory) = self.calculate_box_trajectory()
                elif TRAJECTORY is 2:
                    (self.position_trajectory, self.time_trajectory,
                     self.yaw_trajectory) = self.calculate_fig8_trajectory()
                elif TRAJECTORY is 3:
                    (self.position_trajectory, self.time_trajectory,
                     self.yaw_trajectory) = self.calculate_hover()
                elif TRAJECTORY is 4:
                    (self.position_trajectory, self.time_trajectory,
                     self.yaw_trajectory) = self.calculate_diagonal_trajectory()
                elif TRAJECTORY is 5:
                    (self.position_trajectory, self.time_trajectory,
                     self.yaw_trajectory) = self.calculate_leftright_trajectory()
                else: # otherwise, perform the default test trajectory
                    print("Loading Test Trajectory...")
                    (self.position_trajectory, self.time_trajectory,
                     self.yaw_trajectory) = self.load_test_trajectory(time_mult=0.5)
                self.all_waypoints = self.position_trajectory.copy()
                self.waypoint_number = -1
                self.waypoint_transition()
                print("Begin flight...")
        elif self.flight_state == States.WAYPOINT:
            if time.time() > self.time_trajectory[self.waypoint_number]:
                if len(self.all_waypoints) > 0:
                    if time.time()-self.pos_time > 0.04: # pace the calls to 25fps
                        self.waypoint_transition()
                        self.pos_time = time.time()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self): # 80 frames per second
        if time.time()-self.prev_time > 0.05: # pace the calls to 20fps
            if self.flight_state == States.LANDING:
                if self.global_position[2] - self.global_home[2] < 0.1:
                    if abs(self.local_position[2]) < 0.01:
                        self.disarming_transition()
            if self.flight_state == States.WAYPOINT:
                self.position_controller()

    def state_callback(self): # 1 frame per second
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def calculate_fig8_trajectory(self):
        print("Creating Figure 8 Trajectory...") # same distance and timing as test trajectory
        total_time = 19.4
        t = np.linspace(0.0, total_time, int(total_time / 0.05)) # inspired by class exercise
        xpath = 7.7186622 * np.sin(0.323875531 * (t+2.35) * 2) - 7.70955541
        ypath = 7.7186622 * np.cos(0.323875531 * (t+2.35)) - 7.7186622 + 2.12979
        zpath = np.cos(0.323875531 * (t+2.35)) - 4.0 + 0.2759274
        position_trajectory = []
        for n in range(0, len(xpath)):
            position_trajectory.append(np.array([xpath[n], ypath[n], zpath[n]]))
        yaw_trajectory = []
        for i in range(0, len(position_trajectory)-1):
            yaw_trajectory.append(np.arctan2(position_trajectory[i+1][1]-position_trajectory[i][1], position_trajectory[i+1][0]-position_trajectory[i][0]))
        yaw_trajectory.append(yaw_trajectory[-1])
        current_time = time.time() #+ 0.18
        time_trajectory = np.linspace(current_time, current_time+total_time, int(total_time/0.05))
        return(position_trajectory, time_trajectory, yaw_trajectory)

    def calculate_leftright_trajectory(self):
        print("Creating Left-Right Trajectory...") # same distance and timing as test trajectory
        position_trajectory = [self.local_position]
        for i in range(12):
            position_trajectory.append(np.array([0.0,  1.5, -3.5]))
            position_trajectory.append(np.array([0.0, -1.5, -3.5]))
        position_trajectory.append(np.array([0.0,  1.5, -3.5]))
        position_trajectory.append(np.array([0.0,  0.0, -3.0]))
        current_time = time.time()
        time_trajectory = []
        for i in range(27):
            time_trajectory.append(current_time+0.18+i*0.746154)
        yaw_trajectory = []
        for i in range(0, len(position_trajectory)):
            yaw_trajectory.append(0.0)
        return(position_trajectory, time_trajectory, yaw_trajectory)

    def calculate_box_trajectory(self):
        print("Creating Box Trajectory...") # same distance and timing as test trajectory
        position_trajectory = [self.local_position, np.array([18.195,    0.0, -4.0]),
                                                    np.array([18.195, 18.195, -5.0]), 
                                                    np.array([   0.0, 18.195, -4.0]),
                                                    np.array([   0.0,    0.0, -3.0])]
        current_time = time.time()
        time_trajectory = [current_time+0.18, current_time+4.85+0.18, current_time+9.7+0.18, current_time+14.55+0.18, current_time+19.4+0.18]
        yaw_trajectory = []
        for i in range(0, len(position_trajectory)-1):
            yaw_trajectory.append(np.arctan2(position_trajectory[i+1][1]-position_trajectory[i][1], position_trajectory[i+1][0]-position_trajectory[i][0]))
        yaw_trajectory.append(yaw_trajectory[-1])
        return(position_trajectory, time_trajectory, yaw_trajectory)

    def calculate_diagonal_trajectory(self):
        print("Creating Box Trajectory...") # same distance and timing as test trajectory
        position_trajectory = [self.local_position, np.array([57.2, 45.0, -3.0])]
        current_time = time.time()
        time_trajectory = [current_time+0.18, current_time+19.4+0.18]
        yaw_trajectory = []
        for i in range(0, len(position_trajectory)-1):
            yaw_trajectory.append(np.arctan2(position_trajectory[i+1][1]-position_trajectory[i][1], position_trajectory[i+1][0]-position_trajectory[i][0]))
        yaw_trajectory.append(yaw_trajectory[-1])
        return(position_trajectory, time_trajectory, yaw_trajectory)

    def calculate_hover(self):
        print("Creating hover trajectory...") # same timing as test trajectory
        position_trajectory = [np.array([0.0, 0.0, -3.0]), np.array([0.0, 0.0, -3.0])] #[self.local_position, self.local_position]
        current_time = time.time()
        time_trajectory = [current_time+0.18, current_time+19.4+0.18]
        yaw_trajectory = []
        for i in range(0, len(position_trajectory)-1):
            yaw_trajectory.append(np.arctan2(position_trajectory[i+1][1]-position_trajectory[i][1], position_trajectory[i+1][0]-position_trajectory[i][0]))
        yaw_trajectory.append(yaw_trajectory[-1])
        return(position_trajectory, time_trajectory, yaw_trajectory)

    def arming_transition(self):
        print("arming transition")
        self.take_control()
        self.arm()
        # set the current location to be the home position
        self.set_home_position(self.global_position[0],
                               self.global_position[1],
                               self.global_position[2])  

        self.flight_state = States.ARMING

    def takeoff_transition(self):
        print("takeoff transition")
        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)
        self.flight_state = States.TAKEOFF

    def waypoint_transition(self):
        self.waypoint_number = self.waypoint_number + 1
        self.target_position = self.all_waypoints.pop(0)
        #self.target_position = np.array([0.0, 0.0, -3.0]) # #######
        #print("Waypoint transition:", self.waypoint_number, self.target_position)
        self.flight_state = States.WAYPOINT

    def landing_transition(self):
        print("landing transition")
        self.land()
        self.flight_state = States.LANDING

    def disarming_transition(self):
        print("disarm transition")
        self.disarm()
        self.release_control()
        self.flight_state = States.DISARMING

    def manual_transition(self):
        print("manual transition")
        self.stop()
        self.in_mission = False
        self.flight_state = States.MANUAL

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        # self.connect()

        print("starting connection")
        # self.connection.start()

        super().start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    print("+- Select a mission ----------------------------+")
    print("|  0 (or Enter) for the test trajectory         |")
    print("|  1 for the square box                         |")
    print("|  2 for the figure-8                           |")
    print("|  3 for hover                                  |")
    print("|  4 for diagonal                               |")
    print("|  5 for left-right                             |")
    print("+-----------------------------------------------+")
    traj = input("Select: ")
    if traj is "1": TRAJECTORY = 1
    if traj is "2": TRAJECTORY = 2
    if traj is "3": TRAJECTORY = 3
    if traj is "4": TRAJECTORY = 4
    if traj is "5": TRAJECTORY = 5
    if TRAJECTORY is 0: print("Mission: follow the test trajectory")
    if TRAJECTORY is 1: print("Mission: follow the box trajectory")
    if TRAJECTORY is 2: print("Mission: follow the figure-8 trajectory")
    if TRAJECTORY is 3: print("Mission: hover in place")
    if TRAJECTORY is 4: print("Mission: follow the diagonal trajectory")
    if TRAJECTORY is 5: print("Mission: roll left-right 3 meters")
    conn = MavlinkConnection('tcp:127.0.0.1:5760', threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://127.0.0.1:5760')
    drone = ControlsFlyer(conn)
    time.sleep(2)
    drone.start()
    drone.print_mission_score()
