"""
Nonlinear Controller
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.55                 # kg
GRAVITY = -9.80665                   # m/s^2
MOI = np.array([0.005, 0.005, 0.01]) # kg*m^2
MIN_THRUST = 0.1
MAX_THRUST = 10.0                    # N  = kg*m/s^2
MAX_TORQUE = 1.0                     # Nm = kg*m^2/s^2

class NonlinearController(object):

    def __init__(self):

        # Scalar control gains (similar order & naming as C++ project)
        # Position
        self.kpPosXY = 1.38653
        self.kpPosZ  = 5.0
        self.kiPosXY = 0.25

        # Velocity
        self.kpVelXY = 3.0
        self.kpVelZ  = 1.5
        self.kpAccFF = 0.0577

        # Angle
        self.kpBank = 8.0
        self.kpYaw  = 1.5

        # Angle rate
        self.kpPQR = np.array([20.0, 20.0, 5.0])

        # Limits (min lat vel 3.65 m/s required to complete test trajectory)
        self.maxAccelXY     = 12.0 # m/s^2
        self.maxAscentRate  =  5.0 # m/s
        self.maxDescentRate =  2.0 # m/s
        self.maxTiltAngle   =  1.0 # radians

        # Error state storage
        self.prev_vel = np.array([0.0, 0.0, 0.0])
        self.iError = np.array([0.0, 0.0])

    def trajectory_control(self, position_trajectory,
                           yaw_trajectory,
                           time_trajectory,
                           current_time,
                           return_accff=False):
        """Generate a commanded position, velocity, and yaw based on
            the trajectory
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list of yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the
                            position and yaw commands
            current_time: float corresponding to the current time in seconds
        Returns: command tuple (position, velocity, yaw, accelff)
        """
        ind_min = np.argmin(np.abs(np.array(time_trajectory)-current_time))
        time_ref = time_trajectory[ind_min]
        if current_time < time_ref:
            position0 = position_trajectory[ind_min-1]
            position1 = position_trajectory[ind_min]
            time0 = time_trajectory[ind_min-1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min-1]
        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory)-1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                time0 = 0.0
                time1 = 1.0
            else:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min+1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min+1]
        position_cmd = (position1-position0)* \
                        (current_time-time0)/(time1-time0)+position0
        velocity_cmd = (position1-position0)/(time1-time0)
        accel_cmd = (velocity_cmd - self.prev_vel)/(time1-time0)
        self.prev_vel = velocity_cmd
        if return_accff:
            return (position_cmd, velocity_cmd, yaw_cmd, accel_cmd)
        else:
            return (position_cmd, velocity_cmd, yaw_cmd)


    # #########################################################################
    # BODY RATE CONTROL

    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame
        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) 
                        in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2
            
        Returns: 3-element numpy array, desired roll moment, pitch moment,
                and yaw moment commands in newton*meters
        """
        br_error = body_rate_cmd - body_rate

        # Account for the moment of inertia of the drone
        # kg*m^2 * rad/sec^2 = kg m^2 / sec^2
        momentCmd = self.kpPQR * br_error * MOI
        # Gate the moment if necessary
        if np.linalg.norm(momentCmd) > MAX_TORQUE:
            momentCmd = momentCmd * MAX_TORQUE / np.linalg.norm(momentCmd)
        return momentCmd


    # #########################################################################
    # ATTITUDE CONTROL

    def altitude_control(self, altitude_cmd,
                         vertical_velocity_cmd,
                         altitude,
                         vertical_velocity,
                         attitude,
                         acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command
        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: 3-element numpy array (roll,pitch,yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)
        Returns: thrust command for the vehicle (+up)
        """
        vertical_velocity_cmd += self.kpPosZ * (altitude_cmd - altitude)
        # Gate the ascent/descent rates
        vertical_velocity_cmd = np.clip(vertical_velocity_cmd, -self.maxDescentRate, self.maxAscentRate)
        z_err_dot = vertical_velocity_cmd - vertical_velocity
        u_1_bar = self.kpVelZ * z_err_dot + acceleration_ff

        # Account for drone mass and non-linear effects from non-zero roll/pitch angles
        thrust = DRONE_MASS_KG * u_1_bar / (np.cos(attitude[0]) * np.cos(attitude[1])) # R[2,2]
        # Gate the thrust if necessary
        return np.clip(thrust, MIN_THRUST, MAX_THRUST)

    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        #acceleration_cmd[0] = 0.0
        #acceleration_cmd[1] = 0.0
        """ Generate the rollrate and pitchrate commands in the body frame
        Args:
            acceleration_cmd: 2-element numpy array 
                (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll,pitch,yaw) in radians
            thrust_cmd: vehicle thrust command in newtons
        Returns: 2-element numpy array, desired rollrate (p) and 
                pitchrate (q) commands in radians/s
        """
        # Calculate rotation matrix
        R = euler2RM(attitude[0], attitude[1], attitude[2])
        # Account for drone mass (factor out for acceleration)
        accel = thrust_cmd / DRONE_MASS_KG  # N/kg -> m/s^2
        # Gate the tilt angle as a percentage of total acceleration (approximately)
        bx_targ = -np.clip(acceleration_cmd[0]/accel, -self.maxTiltAngle, self.maxTiltAngle)
        by_targ = -np.clip(acceleration_cmd[1]/accel, -self.maxTiltAngle, self.maxTiltAngle)
        # Calculate angular velocity
        bxd = self.kpBank * (bx_targ - R[0,2]);
        byd = self.kpBank * (by_targ - R[1,2]);
        p_cmd = (R[1,0] * bxd - R[1,1] * byd) / R[2,2]
        q_cmd = (R[0,0] * bxd - R[0,1] * byd) / R[2,2]
        return np.array([p_cmd, q_cmd])

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate
        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians
        Returns: target yawrate in radians/sec
        """
        # Constrain to valid yaw range of -pi to pi
        yaw_error = yaw_cmd - yaw
        if yaw_error > np.pi:
            yaw_error = yaw_error - 2.0 * np.pi
        elif yaw_error < -np.pi:
            yaw_error = yaw_error + 2.0 * np.pi
        return self.kpYaw * yaw_error


    # #########################################################################
    # POSITION CONTROL

    def lateral_position_control(self, local_position_cmd,
                         local_velocity_cmd, local_position, local_velocity,
                         acceleration_ff = np.array([0.0, 0.0]), dt=0.025):
        """Generate horizontal acceleration commands for the vehicle in the
           local frame
        Args:
            local_position_cmd: desired 2D position in local frame 
                                [north, east]
            local_velocity_cmd: desired 2D velocity in local frame 
                                [north_velocity, east_velocity]
            local_position: vehicle position in the local frame 
                            [north, east]
            local_velocity: vehicle velocity in the local frame
                            [north_velocity, east_velocity]
            acceleration_ff: feedforward acceleration command
        Returns: desired vehicle 2D vector in the local frame [north, east]
        """
        local_pos_err = self.kpPosXY * (local_position_cmd - local_position) # m/s
        if np.linalg.norm(self.iError) < 0.25:
            self.iError += local_pos_err * dt
        # Compute acceleration
        vector = self.kpAccFF*acceleration_ff + local_pos_err + \
                 self.kpVelXY * (local_velocity_cmd - local_velocity) + self.kiPosXY * self.iError
        # Constrain computed acceleration
        dist = np.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
        if dist > self.maxAccelXY:
            norm_acc = vector / dist
            vector = norm_acc * self.maxAccelXY
        return vector
