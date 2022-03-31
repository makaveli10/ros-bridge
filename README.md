# ROS/ROS2 bridge for CARLA simulator

[![Actions Status](https://github.com/carla-simulator/ros-bridge/workflows/CI/badge.svg)](https://github.com/carla-simulator/ros-bridge)
[![Documentation](https://readthedocs.org/projects/carla/badge/?version=latest)](http://carla.readthedocs.io)
[![GitHub](https://img.shields.io/github/license/carla-simulator/ros-bridge)](https://github.com/carla-simulator/ros-bridge/blob/master/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/carla-simulator/ros-bridge)](https://github.com/carla-simulator/ros-bridge/releases/latest)

 This ROS package is a bridge that enables two-way communication between ROS and CARLA. The information from the CARLA server is translated to ROS topics. In the same way, the messages sent between nodes in ROS get translated to commands to be applied in CARLA.



**This version requires CARLA 0.9.12**

## Features
- Provide Sensor Data (Lidar, Semantic lidar, Cameras (depth, segmentation, rgb, dvs), GNSS, Radar, IMU)


## Getting started and documentation

### Docker setup
- Clone the ros-bridge repo with carla_msgs submodule.
```bash
 git clone --recurse-submodules -b clean_ros_bridge https://github.com/makaveli10/ros-bridge.git
 cd ros-bridge/
```

- Build docker image
```bash
 docker build -t carlafox .
```


- Start the Carla Server
```bash
 ./CarlaUE4.sh -RenderOffScreen -nosound 
```


- Start docker container with Foxglove web interface
```bash
 docker run -it -d -p 9090:9090 -p 8080:8080 carlafox 
```
NOTE: Use Chrome browser -> localhost:8080. Use Open Connection -> Rosbridge (ROS1 & ROS2). You can user foxglove_layout.json from this repo.


- Create another docker bash terminal, setup environment and run example
```bash
 docker exec -it "container_id" bash
 cd /opt/carla-ros-bridge
 source ./devel/setup.bash
 roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch
```
NOTE: Please check the map topic in 3D panel to see the map.


- To run in passive mode where rosbridge won't be ticking but only publishing data
```bash
 roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch passive:=True
```
NOTE: Another client must tick otherwise carla-ros-bridge will freeze.



