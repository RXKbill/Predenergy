#!/bin/bash
sudo chmod 777 /dev/ttyTHS1

source ../nlink_parser/devel/setup.bash
roslaunch nlink_parser linktrack.launch & sleep 3;
roslaunch time_synchronization time_synchronization.launch;

wait;