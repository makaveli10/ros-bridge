#!/bin/bash
set -e

# install bevfusion
cd /opt/carla-ros-bridge/src/carla_ros_bridge/src/bevfusion
python3 setup.py develop

# setup ros environment
source "/opt/carla-ros-bridge/devel/setup.bash"

python3 -u /wait-for-carla.py "${CARLA_HOSTNAME:-carla}"

exec roslaunch carla_ros_bridge carla_ros_bridge.launch \
  host:="${CARLA_HOSTNAME:-carla}" \
  ${ROSBRIDGE_WEBSOCKET_EXTERNAL_PORT:+"websocket_external_port:=${ROSBRIDGE_WEBSOCKET_EXTERNAL_PORT}"}
