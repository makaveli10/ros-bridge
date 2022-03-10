
ARG ROS_DISTRO

FROM ros:$ROS_DISTRO-ros-base

# install foxglove studio & dependencies
RUN apt update && apt install -y curl wget git git-lfs debian-keyring debian-archive-keyring apt-transport-https
RUN curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash - && \
    apt install nodejs

RUN git clone -b v1.2.0 https://github.com/foxglove/studio.git
RUN curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | tee /etc/apt/trusted.gpg.d/caddy-stable.asc && \
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list && \
        apt update && \
        apt install caddy

RUN cd studio && npm install -g yarn && \
                 yarn install --immutable && \
                 yarn run web:build:prod && \
                 cp -r /studio/web/.webpack/* ./

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

WORKDIR /studio

CMD ["caddy", "file-server", "--listen", ":8080"]