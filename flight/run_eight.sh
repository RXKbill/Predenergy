#!/bin/bash
source ./px4_ctrl/devel/setup.bash

roslaunch px4ctrl run_ctrl.launch & sleep 1;
roslaunch trajectory traj_gen.launch & sleep 1;
roslaunch traj_server traj_server.launch & sleep 1;

wait;