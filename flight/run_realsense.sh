#!/bin/bash

source ../realsense_ros/devel/setup.bash
roslaunch realsense2_camera rs_vins_d430.launch;
wait;