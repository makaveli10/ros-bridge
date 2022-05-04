ARG ROS_DISTRO=noetic

FROM ros:$ROS_DISTRO-ros-base

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
                  if [ "$ROS_VERSION" == "2" ]; then colcon build; else catkin_make install; fi'

# ROS1 patch harmless exception (overflowing logs) :
# Exception calling subscribe callback: 'Clock' object has no attribute ‘_buff’
RUN /bin/bash -c 'if [ "$ROS_VERSION" == "1" ]; then patch /opt/ros/noetic/lib/python3/dist-packages/rosbridge_library/internal/subscribers.py \
                  /opt/carla-ros-bridge/src/patchfile.patch; \
                  else : ; fi'

COPY ./config/default /etc/nginx/sites-enabled

WORKDIR /studio

COPY start-ros-bridge.sh wait-for-carla.py /
ENTRYPOINT ["/start-ros-bridge.sh"]
