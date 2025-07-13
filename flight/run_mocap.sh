#!/bin/bash
source /home/fast/xinweicc_ws/nokov_ws/devel/setup.bash

roslaunch vrpn_client_ros sample.launch server:=10.1.1.198 & sleep 3;
roslaunch ekf nokov.launch & sleep 3;
wait;