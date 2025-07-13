#!/bin/bash

source ../bridge_ws/swarm_bridge/devel/setup.bash
roslaunch swarm_bridge bridge_tcp.launch;
wait;