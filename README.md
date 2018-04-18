# FCND - Controls Project

![C++ trajectory](images/cover.png?raw=true)

## Overview

The goal of the project is to first implement a prototype controller in Python and then translate that code into C++ with some modifications that will improve its robustness and performance. 
<br><br><br>

## Associated files

Program, script, configuration, and log:<br>
**Python script:** `controller.py`<br>
**Log file:** `Logs/TLog-20180417.2007.txt`<br><br>
Located in: https://github.com/jwdunn1/FCND-Controls-CPP:<br>
**C++ program:** `QuadController.cpp`<br>
**Configuration:** `QuadControlParams.txt`
<br><br><br>

## Contents

The following report consists of 4 sections:

**01 Implementation**<br>
    01.1 Body rate control<br>
    01.2 Altitude control<br>
    01.3 Roll-pitch control<br>
    01.4 Yaw control<br>
    01.5 Lateral position control<br>
    01.6 Motor commands in C++

**02 Flight evaluation**<br>
    02.1 Python controller performance<br>
    02.2 C++ controller performance

**03 References**<br>
    Books, research papers, and tools

**04 Appendix**<br>
    04.1 Rotation matrix<br>
    04.2 Angular velocity<br>
    04.3 Thrust computation
<br><br><br>

## 01 Implementation

Each of the implemented methods of the architecture fit together as illustrated in Figure 1. Beginning with the code from the `Full 3D Control` exercise presented in class, the methods were refactored several times to gain a full understanding of the mathematics, physics, and data flow.

![Architecture](images/Figure1.png?raw=true)<br>
Figure 1: Control structure

## 



### 01.1 Body rate control

The controller is a proportional controller on body rates to commanded moments. The controller takes into account the moments of inertia of the drone when calculating the commanded moments. For the Python version, this controller operates at a frequency of 40 hertz through the gyro callback. (The C++ version operates all controllers synchronously at 500 Hz.)

Python: [see lines 105-112 in `controller.py`]<br>
C++: [see lines 111-117 in `QuadController.cpp`]

## 
### 01.2 Altitude control

Part one of attitude control, the altitude controller uses both the down position and the down velocity to command thrust. The drone's mass is accounted for to ensure that the output value is a thrust value in newtons. The thrust includes the non-linear effects from non-zero roll/pitch angles.

Note: The Python version of the attitude controller (which includes altitude, roll-pitch, and yaw controllers) operates at 40 Hz through the attitude callback.

Python: [see lines 135-144 in `controller.py`]

Additionally, the C++ altitude controller contains an integrator to handle the weight non-idealities presented in scenario 4.

C++: [see lines 149-159 in `QuadController.cpp`]

## 
### 01.3 Roll pitch control

Part two of attitude control, the roll-pitch controller uses the acceleration and thrust commands, in addition to the vehicle attitude to output a body rate command. The controller accounts for the non-linear transformation from local accelerations to body rates. The drone's mass is accounted for when calculating the target angles. See also the rotation matrix and angular velocity in the appendix below. Lateral acceleration is limited by a maximum tilt angle.

Python: [see lines 158-170 in `controller.py`]<br>
C++: [see lines 190-199 in `QuadController.cpp`]

## 
### 01.4 Yaw control

Part three of attitude control, the yaw controller is a linear/proportional heading mechanism which outputs yaw rate commands. Further, the yaw error is checked for a valid range between -π and π.

Python: [see lines 179-185 in `controller.py`]<br>
C++: [see lines 222-228 in `QuadController.cpp`]

## 
### 01.5 Lateral position control

Finally, the lateral position controller uses the local NE position and velocity to generate a commanded local acceleration. The Python version operates at 80 Hz through the velocity callback. The trajectory_control  routine optionally returns an acceleration feed-forward value to pass along to the lateral position controller (a testing script mentioned in section 07 below makes use of this feed forward value). Additionally, to smooth the sharp corners of the test trajectory and improve computation of the commanded acceleration, an integrator uses the time duration since the last call.

Python: [see lines 208-219 in `controller.py`]<br>
C++: [see lines 263-274 in `QuadController.cpp`]

## 
### 01.6 Motor commands in C++.

The thrust and moments are converted to the appropriate four different desired thrust commands for the motors. The dimensions of the drone arm length (`L`) and motor torque coefficient (`kappa`) are accounted for when calculating thrust from desired rotation moments. See also the thrust computation in the appendix below.

C++: [see lines 77-86 in `QuadController.cpp`]
<br><br><br>


## 02 Flight evaluation

### 02.1 Python controller performance
The Python controller successfully follows the provided test trajectory, meeting the minimum flight performance metrics. See the log file (`TLog.txt`) in the `Logs` folder. The control gains were discovered manually through trial and error.

For this, the drone passes the provided evaluation script with the default parameters. These metrics being: 

1. the drone flies the test trajectory faster than 20 seconds
2. the maximum horizontal error is less than 2 meters
3. the maximum vertical error is less than 1 meter

Figures 2-4 below are graphic results from a successful flight.

![Test results](images/Figure2.png?raw=true)<br>
Figure 2: Example results from the test script.

## 

![Test trajectory](images/Figure3.png?raw=true)<br>
Figure 3: Overhead view of test trajectory (gray line), flight path (dashed blue), start (green), goal (red). The light gray area is a 2-meter lateral tolerance.

## 

![Test trajectory](images/Figure4.png?raw=true)<br>
Figure 4: Horizontal and vertical error metrics from a successful flight

The NonlinearController class can operate with the default controls_flyer.py script, or with the controls_flyer-TEST.py script. The `TEST` version improves performance by slowing the attitude and position controllers to 20 Hz. A comparison sequence of 20 successful runs of each script is plotted in Figure 5 and demonstrates vertical error reduction by 31.5% (blue) and horizontal error reduction by 2.3% (red). The `TEST` version also includes alternative trajectories useful for tuning the control gains. A hover stability test indicates the Unity simulator contains a GPS noise radius less than 0.25 meter.

![Python test runs](images/Figure5.png?raw=true)<br>
Figure 5: Error metrics from 40 successful flights, comparing the default script (dashed) with reduced rate script (solid). Vertical error is blue and horizontal error is red. Lower values are better.

## 
### 02.2 C++ controller performance
The C++ controller successfully follows the provided test trajectory and passes (numerically and visually) all scenarios. See Figures 6-11 below.

In each scenario, the drone looks stable and performs the required task. The controller is able to handle the non-linearities of scenario 4 (all three drones in the scenario perform their required tasks with the same control gains). The control gains were discovered manually through trial and error, finding workable ranges and setting to mid-points, except yaw gain which is intentionally set to a lower value for higher stability.

![Passing all scenarios](images/Figure6.png?raw=true)<br>
Figure 6: Metrics from successful scenarios (note all are `PASS`)

## 

![Scenario 1](images/Figure7.png?raw=true)<br>
Figure 7: Scenario 1 - tuning mass to hover for at least 0.8 seconds

## 

![Scenario 2](images/Figure8.png?raw=true)<br>
Figure 8: Scenario 2 - stabilize the rotational motion and bring the vehicle back to level attitude

## 

![Scenario 3](images/Figure9.png?raw=true)<br>
Figure 9: Scenario 3 - move to a destination (yellow yaws in flight)

## 

![Scenario 4](images/Figure10.png?raw=true)<br>
Figure 10: Scenario 4 - nonidealities and robustness (red: overweight, orange: ideal, green: shifted mass)

## 

![Scenario 5](images/Figure11.png?raw=true)<br>
Figure 11: Scenario 5 - trajectory following (red: no feed-forward, orange: with feed-forward)

An additional scenario tests hover stability. In this case, position following error is less than 1.2 millimeters for at least 3.33 seconds.
<br><br><br>

## 03 References

[1] ***Quad Rotorcraft Control***<br>
Carrillo, Lopez, Lozano, and Pegard<br>
https://www.springer.com/us/book/9781447143987

[2] ***The GRASP Multiple Micro UAV Testbed***<br>
Michael, Mellinger, Lindsey, and Kumar<br>
https://pdfs.semanticscholar.org/20b0/f0268bc11c55389816223d712d85203e2936.pdf

[3] ***Feedback Systems, 2nd Ed***<br>
Karl J. Åström and Richard M. Murray<br>
http://www.cds.caltech.edu/~murray/amwiki/index.php

[4] ***Feed-Forward Parameter Identification for Precise Periodic Quadrocopter Motions***<br>
Schoellig, Wiltsche, and D’Andrea<br>
http://www.dynsyslab.org/wp-content/papercite-data/pdf/schoellig-acc12.pdf

[5] ***On-board Model Predictive Control of a Quadrotor Helicopter***<br>
Patrick Bouffard<br>
https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-241.pdf

[6] ***Quadcopter Dynamics and Simulation***<br>
Andrew Gibiansky<br>
http://andrew.gibiansky.com/blog/physics/quadcopter-dynamics

[7] ***Propeller Thrust and Drag in Forward Flight***<br>
Rajan Gill and Raffaello D’Andrea<br>
http://flyingmachinearena.org/wp-content/publications/2017/gilIEEE17.pdf

[8] ***Thrust Mixing, Saturation, and Body-Rate Control***<br>
Faessler, Falanga, and Scaramuzza<br>
http://rpg.ifi.uzh.ch/docs/RAL17_Faessler.pdf

[9] ***Modelling and Control of the Crazyflie Quadrotor***<br>
Marcus Greiff<br>
http://lup.lub.lu.se/student-papers/record/8905295/file/8905299.pdf

[10] ***Control of VTOL Vehicles with Thrust-direction Tilting***<br>
Hua, Hamel, and Samson<br>
https://arxiv.org/pdf/1308.0191.pdf

[11] ***A platform for aerial robotics research and demonstration: The Flying Machine Arena***<br>
Lupashin, Hehn, Mueller, Schoellig, Sherback, and D’Andrea<br>
http://flyingmachinearena.org/wp-content/publications/2014/lupashin2014platform.pdf

[12] ***Matrix Equations Solver***<br>
https://www.symbolab.com/solver/matrix-equations-calculator

[13] ***Servo Tuning***<br>
http://support.motioneng.com/downloads-notes/tuning
<br><br><br>

## 04 Appendix

### 04.1 Rotation matrix

![Rotation matrix](images/rotMatrix.png?raw=true)<br><br>

## 
### 04.2 Angular velocity

![Angular velocity](images/angularVelocity.png?raw=true)

## 
### 04.3 Thrust computation

![Thrust compute](images/thrusts.png?raw=true)

![C++ trajectory](images/Figure12.png?raw=true)<br>
Figure 12: Body coordinates and forces
<br><br><br>

    End of report