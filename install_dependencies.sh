#!/usr/bin/env bash

PYTHON_SUFFIX="3"

if [ "$ROS_VERSION" = "2" ]; then
    ADDITIONAL_PACKAGES="ros-$ROS_DISTRO-rviz2"
else
    ADDITIONAL_PACKAGES="ros-$ROS_DISTRO-rviz
                         ros-$ROS_DISTRO-opencv-apps
                         ros-$ROS_DISTRO-rospy-message-converter
                         ros-$ROS_DISTRO-pcl-ros"
fi

echo ADDITIONAL PACKAGES $ADDITIONAL_PACKAGES

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
    ros-$ROS_DISTRO-derived-object-msgs \
    ros-$ROS_DISTRO-rosbridge-suite \
    ros-$ROS_DISTRO-cv-bridge \
    python3-tk \
    wget \
    qt5-default \
    ros-$ROS_DISTRO-ackermann-msgs \
    ros-$ROS_DISTRO-pcl-conversions \
    build-essential \
    ros-$ROS_DISTRO-rqt-image-view \
    ros-$ROS_DISTRO-rqt-gui-py \
    python-is-python3 \
    $ADDITIONAL_PACKAGES

python3 -m pip install --upgrade pip
python3 -m pip install -r /opt/carla-ros-bridge/requirements.txt