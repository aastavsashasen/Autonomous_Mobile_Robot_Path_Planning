# Autonomous_Mobile_Robot_Path_Planning
Application and Comparison of Potential Fields (PF), Probabilistic Roadmap (PRM) and Rapidly Expanding Random Trees (RRT) for mobile robot path planning

All algorithms are evaluated with the same environment, with the robot’s controller for the PRM and RRT algorithms being the same, and differing for the PF algorithm. The environment is given as a Gridworld map of size 10X10 (m) and resolution 0.1 (m). Note that this environment includes obstacles. For all 3 of the problems, the following system assumptions were made:
1) The robot is a non-holonomic robot with 2 inputs (forward velocity, angular velocity) and 3 states (x position, y position, heading angle).
No slippage occurs between the robot’s wheels and the ground.
2) The time-step (dt) is infinitesimally small such that the simulation approaches the real world scenario, ie) model is sufficiently discretized (by Euler Discretization). This time-step is taken as 0.1 (s) for all simulations.
3) No motion model uncertainty exists, meaning there is no modeling of noise in the robot’s motion model.
4) Similarly, no measurement model uncertainty exists. The measurements are considered to be exactly equal to the robot’s state as well as all obstacle boundaries.
5) Markov chain assumption, meaning that the next state is only dependent on the previous state and the input.
6) The baseline of the robot is taken to be 0.45 (m) in all problems.
7) The robot is considered as a point object on the gridworld map.

## Potential Fields ##

