#!/bin/bash

if [ -e /dev/ttyACM0 ]; then
    DEVICE="/dev/ttyACM0"
else
    DEVICE="/dev/ttyTHS0"
fi

echo "Using device: $DEVICE"

sudo chmod 777 $DEVICE

roslaunch mavros px4.launch fcu_url:="$DEVICE:921600" gcs_url:="udp://@192.168.1.102" & sleep 4;

rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 32 20000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 106 20000 0 0 0 0 0 & sleep 1;

wait;
