#!/usr/bin/env python

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

from matplotlib import cm

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


class LidarToRGB(object):

    """LIDAR point cloud overlay on RGB images implementation.
    """
    
    def __init__(self, dot_extent=2) -> None:
        """
        Constructor

        :param dot_extent: visualization dot extent in pixels (Recomended [1-4]) (default: 2)
        :type dot_extent: int
        """
        self.dot_extent = dot_extent - 1

    def lidar_overlay(self, lidar, lidar_data, im_array, rgb_cam, image_w, image_h):
        """Method to project lidar points on RGB camera image.

        :param lidar_data: carla lidar measurement object
        :type carla_lidar_measurement: carla.LidarMeasurement
        :param im_array: rgb image
        :type im_array: np.ndarray 
        """
        # Get the lidar data and convert it to a numpy array.
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

        # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
        # focus on the 3D points.
        intensity = np.array(p_cloud[:, 3])

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(p_cloud[:, :3]).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

        # This (4, 4) matrix transforms the points from lidar space to world space.
        lidar_2_world = lidar.carla_actor.get_transform().get_matrix()

        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_lidar_points)

        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = np.array(rgb_cam.carla_actor.get_transform().get_inverse_matrix())

        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)

        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y
        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]
        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)

        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        # Finally we can use our calibration matrix to do the actual 3D -> 2D.
        points_2d = np.dot(rgb_cam.carla_actor.calibration, point_in_camera_coords)

        # Normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        color_map = np.array([
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T
        
        if self.dot_extent <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            im_array[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):
                # I'm not a NumPy expert and I don't know how to set bigger dots
                # without using this loop, so if anyone has a better solution,
                # make sure to update this script. Meanwhile, it's fast enough :)
                im_array[
                    v_coord[i]-self.dot_extent : v_coord[i]+self.dot_extent,
                    u_coord[i]-self.dot_extent : u_coord[i]+self.dot_extent] = color_map[i]
        return im_array