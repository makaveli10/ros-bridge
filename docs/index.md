# ROS Bridge Documentation

This is the documentation for the ROS bridge which enables two-way communication between ROS and CARLA. The information from the CARLA server is translated to ROS topics. In the same way, the messages sent between nodes in ROS get translated to commands to be applied in CARLA.

The ROS bridge is compatible with both ROS 1 and ROS 2.

The ROS bridge boasts the following features:

- Provides sensor data for LIDAR, Semantic LIDAR, Cameras (depth, segmentation, rgb, dvs), GNSS, Radar and IMU.
- Provides object data such as transforms, traffic light status, visualisation markers, collision and lane invasion.
- Control of aspects of the CARLA simulation like synchronous mode, playing and pausing the simulation and setting simulation parameters.

---

## Get started

- [__Installing ROS bridge for ROS 1__](ros_installation_ros1.md)
- [__Installing ROS bridge for ROS 2__](ros_installation_ros2.md)

---

## Learn about the main ROS bridge package

- [__CARLA ROS bridge__](run_ros.md) - The main package required to run the ROS bridge
- [__ROS Compatiblity Node__](ros_compatibility.md) - The interface that allows the same API to call either ROS 1 or ROS 2 functions

---

## Learn about the additional ROS bridge packages

- [__CARLA Spawn Objects__](carla_spawn_objects.md) - Provides a generic way to spawn actors
- [__PCL Recorder__](pcl_recorder.md) - Create point cloud maps from data captured from simulations

---

## Explore the reference material

- [__ROS Sensors__](ros_sensors.md) - Reference topics available in the different sensors
- [__ROS messages__](ros_msgs.md) - Reference parameters available in CARLA ROS messages
