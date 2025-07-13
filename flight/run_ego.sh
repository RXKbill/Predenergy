#!/bin/bash
source ./px4_ctrl/devel/setup.bash
roslaunch px4ctrl run_ctrl.launch & sleep 1;

source ./ego_planner/devel/setup.bash
roslaunch ego_planner run_in_exp.launch & sleep 1;

wait;
