#!/usr/bin/env python

#
# Copyright (c) 2022 Collabora Ltd.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
SFA3D TensorRT 3D perception inference  script.
"""

from typing import List, Tuple, Union
import argparse
import sys
import os
import warnings
import time
import math
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa # pylint: disable=unused-import
import pycuda.driver as cuda

import carla_ros_bridge.sfa3d_config as cnf
from carla_ros_bridge.sfa3d_helper import _sigmoid, get_filtered_lidar, decode, \
                        post_processing, Calibration, convert_det_to_real_values, \
                        lidar_to_camera_box, show_rgb_image_with_boxes, makeBEVMap


class FPNResnet18TRT:
    """FPN Resnet18 TensorRT inference utility class.
    """
    def __init__(self, node, engine_path: str=None, onnx_path: str=None):
        """Initialize.

        Args:
            node: CarlaRosBridge object for logging
            engine_path: tensorrt engine file path
            onnx_path: onnx model path
        """
        self.node = node

        # create a Context on this device
        self._ctx = cuda.Device(0).make_context()
        self._logger = trt.Logger(trt.Logger.INFO)
        self._stream = cuda.Stream()

        # initiate engine related class attributes
        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None

        self._load_model(engine_path=engine_path, onnx_path=onnx_path)
        self._allocate_buffers()

    def _preprocess(self, lidar: np.ndarray) -> np.ndarray:
        """Preprocess image and lidar for inference.

        Args:
            lidar: lidar data of shape (N, 4)

        Returns:
            preprocessed lidar bev_map
        """
        # filter lidar data
        lidar = get_filtered_lidar(lidar, cnf.boundary)
        bev_map = makeBEVMap(lidar, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)
        return bev_map

    def _deserialize_engine(self, trt_engine_path: str) -> trt.tensorrt.ICudaEngine:
        """Deserialize TensorRT Cuda Engine

        Args:
            trt_engine_path: path to engine file

        Returns:
            deserialized tensorrt cuda engine
        """
        self.node.loginfo("[INFO] Deserializing TensorRT engine ...")
        with open(trt_engine_path, 'rb') as engine_file:
            with trt.Runtime(self._logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine_file.read())

        return engine

    def _allocate_buffers(self) -> None:
        """Allocates memory for inference using TensorRT engine.
        """
        inputs, outputs, bindings = [], [], []
        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self._engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        # set buffers
        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings

    def _load_model(self, engine_path=None, onnx_path=None) -> None:
        """Deserializes and Loads tensorrt engine.

        Args:
            engine_path: path to the tensorrt engine file
            onnx_path: path to the onnx model

        """
        # build engine with given configs and load it
        if engine_path is None or not os.path.exists(engine_path):
            if onnx_path is None:
                raise FileNotFoundError(f"No ONNX or TensorRT model to load.")
            self._engine = self._build_and_serialize_engine(onnx_path)

        # deserialize and load engine
        if self._engine is None:
            self._engine = self._deserialize_engine(engine_path)
            self.node.loginfo("[INFO] Deserialized TensorRT engine.")
        if not self._engine:
            self.node.logerr("[Error] Couldn't deserialize engine successfully !")

        # create execution context
        self._context = self._engine.create_execution_context()
        if not self._context:
            self.node.logerr(
                "[Error] Couldn't create execution context from engine successfully !")

    def _build_and_serialize_engine(
        self, onnx_path: str, fp16: bool=False) -> trt.tensorrt.ICudaEngine:
        """Builds TensorRT engine from ONNX model, serialized the
        engine and saves it.

        Args:
            onnx_path: path to the onnx model
            fp16: use fp16 quantization

        Raises:
            FileNotFoundError: onnx model to build tensorrt engine is not found

        Returns:
            deserialized tensorrt cuda engine
        """
        # pylint: disable=no-member
        # Checks if onnx path exists.
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"[Error] {onnx_path} does not exists.")

        # TODO: fix this
        trt_engine_path = os.path.join(
            '/'.join(onnx_path.split('/')[:-1]),
            onnx_path.split('/')[-1].replace("onnx", "engine"))

        # Check if onnx_path is valid.
        if ".onnx" not in onnx_path:
            self.node.logerr(
                f"[Error] Expected onnx weight file, instead {onnx_path} is given."
            )

        # specify that the network should be created with an explicit batch dimension
        batch_size = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # build and serialize engine
        with trt.Builder(self._logger) as builder, \
             builder.create_network(batch_size) as network, \
             trt.OnnxParser(network, self._logger) as parser:

            # setup builder config
            config = builder.create_builder_config()
            config.max_workspace_size = 256 *  1 << 20  # 256 MB
            builder.max_batch_size = 1

            # FP16 quantization
            if builder.platform_has_fast_fp16 and fp16:
                trt_engine_path = trt_engine_path.replace('.engine', '_fp16.engine')
                config.flags = 1 << (int)(trt.BuilderFlag.FP16)
            else:
                trt_engine_path = trt_engine_path.replace('.engine', '_fp32.engine')
            if os.path.exists(trt_engine_path):
                self.node.loginfo(f"{trt_engine_path} already exists.")
                return self._deserialize_engine(trt_engine_path)

            # parse onnx model
            with open(onnx_path, 'rb') as onnx_file:
                if not parser.parse(onnx_file.read()):
                    for error in range(parser.num_errors):
                        self.node.logerr(parser.get_error(error))

            # build engine
            engine = builder.build_engine(network, config)
            with open(trt_engine_path, 'wb') as trt_engine_file:
                trt_engine_file.write(engine.serialize())
            self.node.loginfo("[INFO] TensorRT Engine serialized and saved !")
            return engine

    def __call__(
        self,
        img_rgb: np.ndarray,
        lidar: np.ndarray,
        calibs: dict) -> np.ndarray:
        """Runs inference on the given inputs.

        Args:
            img_rgb: rgb image from current carla frame
            lidar: lidar data numpy array
            calibs: calibration data dict with P2 and sensor transforms

        Returns:
            image with 3D bounding boxes
        """
        bev_maps = self._preprocess(lidar)
        inputs = bev_maps.float().contiguous()
        self._ctx.push()

        # copy inputs to input memory
        # without astype gives invalid arg error
        self._inputs[0]['host'] = np.ravel(inputs).astype(np.float32)

        # transfer data to the gpu
        t1 = time.time()
        cuda.memcpy_htod_async(
            self._inputs[0]['device'], self._inputs[0]['host'], self._stream)

        # run inference
        self._context.execute_async_v2(bindings=self._bindings,
                                       stream_handle=self._stream.handle)

        # fetch outputs from gpu
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)
        t2 = time.time()

        # synchronize stream
        self._stream.synchronize()
        self._ctx.pop()

        # post-process
        trt_outputs = [out['host'] for out in self._outputs]
        return self._postprocess(img_rgb, trt_outputs, calibs, bev_maps)

    def _postprocess(
        self,
        img_rgb: np.ndarray,
        trt_outputs: List[np.ndarray],
        calibs: dict,
        bev_maps: np.ndarray,
        draw_preds_on_img: bool=True,
        merge_rgb_bev: bool=False) -> Union[np.ndarray, None]:
        """Postprocess outputs.

        Args:
            img_rgb: rgb image from current carla frame to draw predictions
            trt_outputs: raw tensorrt inference outputs
            calibs: calibration data with P2 and sensor transforms
            bev_maps: lidar preprocessed map
            draw_preds_on_img: whether to draw predictions on the image
            merge_rgb_bev: whether to merge image & lidar map to draw
                           predictions for output image

        Returns:
            output image if detections > 0 else None
        """
        outputs = [torch.from_numpy(output.reshape((1,-1,152,152))) for output in trt_outputs]
        outputs[0] = _sigmoid(outputs[0])
        outputs[1] = _sigmoid(outputs[1])

        detections = decode(outputs[0], outputs[1], outputs[2], outputs[3],
                            outputs[4], K=50)
        detections = detections.numpy().astype(np.float32)
        detections = post_processing(detections)

        detections = detections[0]  # only first batch

        if draw_preds_on_img:
            img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            calib = Calibration(calibs)
            kitti_dets = convert_det_to_real_values(detections.copy())

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(
                    kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                return show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
            if merge_rgb_bev:
                bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
                bev_map = draw_predictions(bev_map, detections.copy())

                # rotate the bev_map
                bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
                return merge_rgb_to_bev(img_bgr, bev_map, output_width=cnf.BEV_WIDTH)
        return None

    def destroy(self):
        """Destroy if any cuda context in the stack.
        """
        try:
            self._ctx.pop()
        except Exception as exception:
            pass
