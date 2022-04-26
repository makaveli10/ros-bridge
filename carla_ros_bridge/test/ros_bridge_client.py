#!/usr/bin/env python
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
Class for testing nodes
"""
# pylint: disable=no-member

import unittest
import rospy
import rostest
from std_msgs.msg import Header, String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, NavSatFix, CompressedImage, PointCloud2, Imu
from geometry_msgs.msg import Quaternion, Vector3, Pose
from nav_msgs.msg import Odometry, Path
from derived_object_msgs.msg import ObjectArray
from visualization_msgs.msg import Marker, MarkerArray
from carla_msgs.msg import (CarlaEgoVehicleStatus, CarlaEgoVehicleInfo, CarlaWorldInfo,
                            CarlaActorList, CarlaTrafficLightStatusList,
                            CarlaTrafficLightInfoList)

PKG = 'test_roslaunch'
TIMEOUT = 150


class TestClock(unittest.TestCase):

    """
    Handles testing of the all nodes
    """

    def test_clock(self):
        """
        Tests clock
        """
        rospy.init_node('test_node', anonymous=True)
        clock_msg = rospy.wait_for_message("/clock", Clock, timeout=TIMEOUT)
        self.assertNotEqual(Clock(), clock_msg)

    def test_vehicle_status(self):
        """
        Tests vehicle_status
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/vehicle_status", CarlaEgoVehicleStatus, timeout=TIMEOUT)
        self.assertNotEqual(msg.header, Header())
        self.assertEqual(msg.header.frame_id, 'map')
        self.assertNotEqual(msg.orientation, Quaternion())

    def test_vehicle_info(self):
        """
        Tests vehicle_info
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo, timeout=TIMEOUT)
        self.assertNotEqual(msg.id, 0)
        self.assertEqual(msg.type, "vehicle.tesla.model3")
        self.assertEqual(msg.rolename, "ego_vehicle")
        self.assertEqual(len(msg.wheels), 4)
        self.assertNotEqual(msg.max_rpm, 0.0)
        self.assertNotEqual(msg.moi, 0.0)
        self.assertNotEqual(msg.damping_rate_full_throttle, 0.0)
        self.assertNotEqual(msg.damping_rate_zero_throttle_clutch_engaged, 0.0)
        self.assertNotEqual(
            msg.damping_rate_zero_throttle_clutch_disengaged, 0.0)
        self.assertTrue(msg.use_gear_autobox)
        self.assertNotEqual(msg.gear_switch_time, 0.0)
        self.assertNotEqual(msg.mass, 0.0)
        self.assertNotEqual(msg.clutch_strength, 0.0)
        self.assertNotEqual(msg.drag_coefficient, 0.0)
        self.assertNotEqual(msg.center_of_mass, Vector3())

    def test_odometry(self):
        """
        Tests Odometry
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/odometry", Odometry, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "map")
        self.assertEqual(msg.child_frame_id, "ego_vehicle")
        self.assertNotEqual(msg.pose, Pose())

    def test_gnss(self):
        """
        Tests Gnss
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/gnss", NavSatFix, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/gnss")
        self.assertNotEqual(msg.latitude, 0.0)
        self.assertNotEqual(msg.longitude, 0.0)
        self.assertNotEqual(msg.altitude, 0.0)

    def test_imu(self):
        """
        Tests IMU sensor node
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message("/carla/ego_vehicle/imu", Imu, timeout=15)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/imu")
        self.assertNotEqual(msg.linear_acceleration, 0.0)
        self.assertNotEqual(msg.angular_velocity, 0.0)
        self.assertNotEqual(msg.orientation, 0.0)

    def test_camera_info(self):
        """
        Tests camera_info
        """
        rospy.init_node('test_node', anonymous=True)
        try:
            msg = rospy.wait_for_message(
                "/carla/ego_vehicle/rgb_front/camera_info", CameraInfo, timeout=TIMEOUT)
        except Exception as e:
            print(e)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/rgb_front")
        self.assertEqual(msg.height, 600)
        self.assertEqual(msg.width, 800)

    def test_camera_image(self):
        """
        Tests camera_images
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/rgb_front/image", CompressedImage, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/rgb_front")
        self.assertNotEqual(msg.data, None)

    def test_third_person_camera_info(self):
        """
        Tests third person camera_info
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/rgb_view/camera_info", CameraInfo, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/rgb_view")
        self.assertEqual(msg.height, 600)
        self.assertEqual(msg.width, 800)

    def test_third_person_camera_image(self):
        """
        Tests third person camera_images
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/rgb_view/image", CompressedImage, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/rgb_view")
        self.assertNotEqual(msg.data, None)
    
    def test_semantic_segmentation_camera_info(self):
        """
        Tests semantic_segmentation camera info
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/semantic_segmentation_front/camera_info", CameraInfo, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/semantic_segmentation_front")
        self.assertEqual(msg.height, 70)
        self.assertEqual(msg.width, 400)
    
    def test_semantic_segmentation_camera_image(self):
        """
        Tests semantic segmentation camera images
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/semantic_segmentation_front/image", CompressedImage, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/semantic_segmentation_front")
        self.assertNotEqual(msg.data, None)
    
    def test_depth_camera_info(self):
        """
        Tests depth camera info
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/depth_front/camera_info", CameraInfo, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/depth_front")
        self.assertEqual(msg.height, 70)
        self.assertEqual(msg.width, 400)

    def test_depth_camera_image(self):
        """
        Tests depth camera images
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/depth_front/image", CompressedImage, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/depth_front")
        self.assertNotEqual(msg.data, None)

    def test_ego_vehicle_path(self):
        """
        Tests ego vehicle travesed path
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/waypoints", Path, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "map")
        self.assertNotEqual(len(msg.poses), 0)

    def test_lidar(self):
        """
        Tests Lidar sensor node
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/lidar", PointCloud2, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/lidar")
        self.assertNotEqual(len(msg.data), 0)

    def test_semantic_lidar(self):
        """
        Tests semantic_lidar sensor node
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/semantic_lidar", PointCloud2, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/semantic_lidar")
        self.assertNotEqual(len(msg.data), 0)

    def test_radar(self):
        """
        Tests Radar sensor node
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/radar_front", PointCloud2, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "ego_vehicle/radar_front")

    def test_ego_vehicle_objects(self):
        """
        Tests objects node for ego_vehicle
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/ego_vehicle/objects", ObjectArray, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "map")
        self.assertEqual(len(msg.objects), 0)

    def test_objects(self):
        """
        Tests carla objects
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message("/carla/objects", ObjectArray, timeout=TIMEOUT)
        self.assertEqual(msg.header.frame_id, "map")
        self.assertEqual(len(msg.objects), 1)  # only ego vehicle exists

    def test_marker(self):
        """
        Tests marker
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message("/carla/markers", MarkerArray, timeout=TIMEOUT)
        self.assertEqual(len(msg.markers), 1)  # only ego vehicle exists

        ego_marker = msg.markers[0]
        self.assertEqual(ego_marker.header.frame_id, "map")
        self.assertNotEqual(ego_marker.id, 0)
        self.assertEqual(ego_marker.type, 1)
        self.assertNotEqual(ego_marker.pose, Pose())
        self.assertNotEqual(ego_marker.scale, Vector3())
        self.assertEqual(ego_marker.color.r, 0.0)
        self.assertEqual(ego_marker.color.g, 255.0)
        self.assertEqual(ego_marker.color.b, 0.0)

    def test_map(self):
        """
        Tests map
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/map", String, timeout=TIMEOUT)
        self.assertNotEqual(len(msg.data), 0)

    def test_world_info(self):
        """
        Tests world_info
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/world_info", CarlaWorldInfo, timeout=TIMEOUT)
        self.assertNotEqual(len(msg.map_name), 0)
        self.assertNotEqual(len(msg.opendrive), 0)

    def test_actor_list(self):
        """
        Tests actor_list
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/actor_list", CarlaActorList, timeout=TIMEOUT)
        self.assertNotEqual(len(msg.actors), 0)

    def test_traffic_lights(self):
        """
        Tests traffic_lights
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/traffic_lights/status", CarlaTrafficLightStatusList, timeout=TIMEOUT)
        self.assertNotEqual(len(msg.traffic_lights), 0)

    def test_traffic_lights_info(self):
        """
        Tests traffic_lights
        """
        rospy.init_node('test_node', anonymous=True)
        msg = rospy.wait_for_message(
            "/carla/traffic_lights/info", CarlaTrafficLightInfoList, timeout=TIMEOUT)
        self.assertNotEqual(len(msg.traffic_lights), 0)


if __name__ == '__main__':
    rostest.rosrun(PKG, 'tests', TestClock)
