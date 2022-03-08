#!/usr/bin/env bash

PYTHON_SUFFIX="3"

sudo apt update
sudo apt-get install --no-install-recommends -y \
    python3-pip \
    python3-rosinstall \
    python3-osrf-pycommon \
    python3-catkin-tools \
    python3-catkin-pkg \
    python3-catkin-pkg-modules \
    python3-rosdep \
    python3-wstool \
    python3-opencv \
    ros-noetic-derived-object-msgs \
    ros-noetic-rosbridge-suite \
    ros-noetic-cv-bridge \
    ros-noetic-ros-numpy \
    python3-tk \
    wget \
    qt5-default \
    ros-noetic-ackermann-msgs \
    ros-noetic-pcl-conversions \
    build-essential \
    ros-noetic-rviz \
    ros-noetic-opencv-apps \
    ros-noetic-rospy-message-converter \
    ros-noetic-pcl-ros \
    ros-noetic-rqt-image-view \
    ros-noetic-rqt-gui-py \
    python-is-python3

python3 -m pip install --upgrade pip
python3 -m pip install -r /carla-ros-bridge/catkin_ws/src/ros-bridge/requirements.txt
