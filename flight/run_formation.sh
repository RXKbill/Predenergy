#!/bin/bash
source ./px4_ctrl/devel/setup.bash
roslaunch px4ctrl run_ctrl.launch & sleep 1;

source ../formation_ws/devel/setup.bash
roslaunch ego_planner run_formation.launch & sleep 1;

wait;
