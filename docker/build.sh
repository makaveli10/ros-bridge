#!/bin/bash

# setup sources.list
if [ "$1" == "noetic" ]; then echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list ; \
    else echo "deb http://packages.ros.org/ros2/ubuntu focal main" > /etc/apt/sources.list.d/ros2-latest.list ; fi

# setup keys
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# ENV ROS_DISTRO noetic

# install ros packages
apt-get update
if [ "$1" == "noetic" ]; then apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* && rm -rf /var/lib/apt/lists/* ; \
    else apt-get install -y --no-install-recommends \
    ros-foxy-ros-core=0.9.2-1* \
    && rm -rf /var/lib/apt/lists/* ; fi

# setup entrypoint
# COPY ./ros_entrypoint.sh /

# install bootstrap tools
if [ "$1" == "noetic" ]; then apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/* ; \
    else apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/* ; fi

# bootstrap rosdep
rosdep init && \
  rosdep update --rosdistro $1

if [ "$1" == "foxy" ]; then colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update ; fi

# install ros packages
if [ "$1" == "noetic" ]; then apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/* ; \
    else apt-get update && apt-get install -y --no-install-recommends \
    ros-foxy-ros-base=0.9.2-1* \
    && rm -rf /var/lib/apt/lists/* ; fi