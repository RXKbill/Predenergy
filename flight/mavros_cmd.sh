#!/bin/bash
rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 83 5000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 105 5000 0 0 0 0 0 & sleep 1;

wait;
