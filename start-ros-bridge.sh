#!/bin/bash
set -e

# setup ros environment
source "/opt/carla-ros-bridge/devel/setup.bash"

python3 -u /wait-for-carla.py carla

exec roslaunch carla_ros_bridge carla_ros_bridge.launch
