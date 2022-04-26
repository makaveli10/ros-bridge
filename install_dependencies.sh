#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SUFFIX="3"

# for map server
if [ "$ROS_VERSION" = "2" ]; then
    ADDITIONAL_PACKAGES="ros-$ROS_DISTRO-navigation2
                         ros-$ROS_DISTRO-nav2-bringup"
else
    ADDITIONAL_PACKAGES="ros-$ROS_DISTRO-map-server
                         ros-$ROS_DISTRO-ros-numpy"
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
    build-essential \
    python-is-python3 \
    $ADDITIONAL_PACKAGES

python3 -m pip install --upgrade pip
python$PYTHON_SUFFIX -m pip install -r $SCRIPT_DIR/requirements.txt
