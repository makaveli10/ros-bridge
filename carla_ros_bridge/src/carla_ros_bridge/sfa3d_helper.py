#!/usr/bin/env python

# original code: https://github.com/maudzung/SFA3D

import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import carla_ros_bridge.sfa3d_config as cnf


class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calibs):
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3, 4])
        
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = self.velo_to_cam(
            calibs['camera_transform'], calibs['lidar_transform'])
        self.V2C = np.reshape(self.V2C, [3, 4])
        
        # Rotation from reference camera coord to rect camera coord
        self.R0 = np.identity(3)
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P2[0, 2]
        self.c_v = self.P2[1, 2]
        self.f_u = self.P2[0, 0]
        self.f_v = self.P2[1, 1]
        self.b_x = self.P2[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P2[1, 3] / (-self.f_v)

    def velo_to_cam(self, camera_transform, lidar_transform):
        # pitch, yaw, roll
        pitch = math.radians(lidar_transform.rotation.pitch - camera_transform.rotation.pitch)
        yaw = math.radians(lidar_transform.rotation.yaw - camera_transform.rotation.yaw)
        roll = math.radians(lidar_transform.rotation.roll - camera_transform.rotation.roll)

        TR = np.array(
            [
                [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)],
                [
                    -math.cos(roll) * math.sin(yaw) + math.sin(roll) * math.sin(pitch) * math.cos(yaw),
                    math.cos(roll) * math.cos(yaw) + math.sin(roll) * math.sin(pitch) * math.sin(yaw),
                    math.sin(roll) * math.cos(pitch),
                ],
                [
                    math.sin(roll) * math.sin(yaw) + math.cos(roll) * math.sin(pitch) * math.cos(yaw),
                    -math.sin(roll) * math.cos(yaw) + math.cos(roll) * math.sin(pitch) * math.sin(yaw),
                    math.cos(roll) * math.cos(pitch),
                ],
            ]
        )
        TR_velodyne = np.dot(TR, np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        TR_velodyne = np.dot(np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]), TR_velodyne)
        TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
        return TR_velodyne


def get_filtered_lidar(lidar, boundary):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
                    (lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
                    (lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
    lidar = lidar[mask]
    lidar[:, 2] = lidar[:, 2] - minZ

    return lidar


def makeBEVMap(PointCloud_, boundary, img_shape=(608, 608)):
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    return RGB_Map


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)

    topk_ys = (torch.div(topk_inds, width, rounding_mode='trunc')).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.div(topk_ind, K, rounding_mode='trunc')).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind)

    topk_inds = topk_inds.view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections


def lidar_to_camera(x, y, z, V2C=None, R0=None, P2=None):
    p = np.array([x, y, z, 1])
    if V2C is None or R0 is None:
        p = np.matmul(cnf.Tr_velo_to_cam, p)
        p = np.matmul(cnf.R0, p)
    else:
        p = np.matmul(V2C, p)
        p = np.matmul(R0, p)
    p = p[0:3]
    return tuple(p)


def lidar_to_camera_box(boxes, V2C=None, R0=None, P2=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(x, y, z, V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi / 2
        # ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)

def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.2):
    """
    Args:
        detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)

    Returns:
        list with post processed outputs
    """
    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,
                detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]
        ret.append(top_preds)

    return ret


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(
        img,
        (corners_int[0, 0], corners_int[0, 1]),
        (corners_int[3, 0], corners_int[3, 1]),
        (255, 255, 0), 2)


def draw_predictions(img, detections, num_classes=3):
    for j in range(num_classes):
        if len(detections[j]) > 0:
            for det in detections[j]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                drawRotatedBox(img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)])
    return img


def convert_det_to_real_values(detections, num_classes=3, add_score=False):
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                z = _z + cnf.boundary['minZ']
                w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                if not add_score:
                    kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw])                    
                else:
                    kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw, _score])
    return np.array(kitti_dets)


def draw_box_3d(image, corners, color=(0, 0, 255)):
    """ Draw 3d bounding box in image
        corners: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image



def show_rgb_image_with_boxes(img, labels, calib):
    for box_idx, label in enumerate(labels):
        cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]
        if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
            continue
        if cls_id < 0:
            continue
        corners_3d = compute_box_3d(dim, location, ry)
        corners_2d = project_to_image(corners_3d, calib.P2)
        img = draw_box_3d(img, corners_2d, color=cnf.colors[int(cls_id)])

    return img


def roty(angle):
    # Rotation about the y-axis.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(dim, location, ry):
    # dim: 3
    # location: 3
    # ry: 1
    # return: 8 x 3
    R = roty(ry)
    h, w, l = dim
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d.astype(np.int)
