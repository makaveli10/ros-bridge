# FROM nvcr.io/nvidia/tensorrt:22.01-py3
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG ROS_DISTRO=noetic
ARG CUDA_VER=cu111

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

COPY ./docker/build.sh /tmp/  
RUN chmod u+x /tmp/build.sh && /tmp/build.sh $ROS_DISTRO

# setup ros bridge
RUN mkdir -p /opt/carla-ros-bridge/src
WORKDIR /opt/carla-ros-bridge/

COPY requirements.txt /opt/carla-ros-bridge/
COPY install_dependencies.sh /opt/carla-ros-bridge/

RUN /bin/bash -c 'source /opt/ros/$ROS_DISTRO/setup.bash; \
                 rosdep update && rosdep install --from-paths src --ignore-src -r'
RUN /bin/bash -c 'source /opt/ros/$ROS_DISTRO/setup.bash; \
                  bash /opt/carla-ros-bridge/install_dependencies.sh;'

COPY . /opt/carla-ros-bridge/src

RUN /bin/bash -c 'source /opt/ros/$ROS_DISTRO/setup.bash ; \
                  if [ "$ROS_DISTRO" == "foxy" ]; then colcon build; else catkin_make install; fi'

# ROS1 patch harmless exception (overflowing logs) :
# Exception calling subscribe callback: 'Clock' object has no attribute ‘_buff’
RUN /bin/bash -c 'if [ "$ROS_DISTRO" == "noetic" ]; then patch /opt/ros/noetic/lib/python3/dist-packages/rosbridge_library/internal/subscribers.py \
                  /opt/carla-ros-bridge/src/patchfile.patch; else : ; fi'

# download bevfusion model
RUN /bin/bash -c 'gdown --fuzzy https://drive.google.com/file/d/1LTlQKbA5Qrkv53P-k_c3UP0xrJVIcn-1/view?usp=sharing \
                 -O /opt/carla-ros-bridge/src/carla_ros_bridge/config/'


COPY ./config/default /etc/nginx/sites-enabled

# WORKDIR /studio

COPY start-ros-bridge.sh wait-for-carla.py /
ENTRYPOINT ["/start-ros-bridge.sh"]
