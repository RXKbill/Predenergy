#!/bin/bash

source ../vins_fusion_gpu/devel/setup.bash
rosrun vins vins_node ../vins_fusion_gpu/src/config/realsense_d430/realsense_stereo_imu_config.yaml
wait;