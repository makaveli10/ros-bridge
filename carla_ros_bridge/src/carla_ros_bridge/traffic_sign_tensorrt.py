import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000
    ]
)

class YOLOv7TRT(object):
    def __init__(
        self, node, engine_path, imgsz=(640,640),
        onnx_path: str=None, fp16: bool=False, stream = None
        ):
        
        self.node = node
        
        # create a Context on this device
        self._ctx = cuda.Device(0).make_context()
        self._logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self._logger,'') # initialize TensorRT plugins
        
        self._stream = cuda.Stream() if stream is None else stream
        self._fp16 = fp16

        # initiate engine related class attributes
        self._engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None
        
        # for building the network
        self._batch_size = None
        self._network = None
        self._parser = None

        self._load_model(engine_path=engine_path, onnx_path=onnx_path)
        self._allocate_buffers()
        
        self.imgsz = imgsz
        self.mean = None
        self.std = None
        self.n_classes = 18
        self.class_names = [ 'Speed Limit 30', 'Speed Limit 30 US', 'Speed Limit 40', 'Speed Limit 60', 'Speed Limit 60 US', 'Speed Limit 90', 'Speed Limit 90 US', 
          'Stop', 'Interchange', 'NoTurnsLeft', 'No Turns', 'One Way', 'One Way Left', 'Michigan Left', 'Lane Reduce Left', 'One Way Right',
          'Yield', 'AnimalCrossing']
            
    def _load_model(self, engine_path, onnx_path):
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

    def _deserialize_engine(self, trt_engine_path: str) -> trt.tensorrt.ICudaEngine:
        """Deserialize TensorRT Cuda Engine

        Args:
            trt_engine_path: path to engine file

        Returns:
            deserialized tensorrt cuda engine
        """
        self.node.loginfo(
            f"[INFO] Deserializing TensorRT engine {trt_engine_path}...")
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
    
    def _build_and_serialize_engine(self, onnx_path, conf_thres=0.5, iou_thres=0.45, max_det=100):
        # create network
        builder = trt.Builder(self._logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 2 * (2 ** 30)   # 2 GB
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self._network = builder.create_network(network_flags)
        self._parser = trt.OnnxParser(self._network, self._logger)

        with open(onnx_path, "rb") as f:
            if not self._parser.parse(f.read()):
                self.node.logerr("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self._parser.num_errors):
                    self.node.logfatal(self._parser.get_error(error))

        inputs = [self._network.get_input(i) for i in range(self._network.num_inputs)]
        outputs = [self._network.get_output(i) for i in range(self._network.num_outputs)]

        self.node.loginfo("Network Description")
        for input in inputs:
            self._batch_size = input.shape[0]
            self.node.loginfo(
                "Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            self.node.loginfo(
                "Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self._batch_size > 0
        builder.max_batch_size = self._batch_size

        previous_output = self._network.get_output(0)
        self._network.unmark_output(previous_output)

        # slice boxes, obj_score, class_scores
        strides = trt.Dims([1,1,1])
        starts = trt.Dims([0,0,0])

        bs, num_boxes, temp = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])

        boxes = self._network.add_slice(previous_output, starts, shapes, strides)
        num_classes = temp -5 
        starts[2] = 4
        shapes[2] = 1

        obj_score = self._network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes

        scores = self._network.add_slice(previous_output, starts, shapes, strides)

        updated_scores = self._network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)

        registry = trt.get_plugin_registry()
        assert(registry)
        creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
        assert(creator)
        fc = []
        fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
        
        fc = trt.PluginFieldCollection(fc) 
        nms_layer = creator.create_plugin("nms_layer", fc)
        layer = self._network.add_plugin_v2([boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
        layer.get_output(0).name = "num"
        layer.get_output(1).name = "boxes"
        layer.get_output(2).name = "scores"
        layer.get_output(3).name = "classes"
        for i in range(4):
            self._network.mark_output(layer.get_output(i))
       
        # create engine
        inputs = [self._network.get_input(i) for i in range(self._network.num_inputs)]

        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if self._fp16:
            if not builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        
        trt_engine_path = onnx_path.replace("onnx", "engine")
        if os.path.exists(trt_engine_path):
            self.node.loginfo(f"{trt_engine_path} already exists.")
            return self._deserialize_engine(trt_engine_path)

        with builder.build_engine(self._network, config) as engine, open(trt_engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(trt_engine_path))
            f.write(engine.serialize())
        return engine

    def _preprocess(self, image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        # if use yolox set 
        # padded_img = padded_img[:, :, ::-1]
        # padded_img /= 255.0
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r           

    def _infer(self, img):
        self._inputs[0]['host'] = np.ravel(img)
        
        # transfer data to the gpu
        for inp in self._inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self._stream)
        
        # run inference
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle)
        
        # fetch outputs from gpu
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)
        
        # synchronize stream
        self._stream.synchronize()

        data = [out['host'] for out in self._outputs]
        return data
    
    def __call__(self, origin_img, conf=0.5):
        img, ratio = self._preprocess(origin_img, self.imgsz, self.mean, self.std)
        data = self._infer(img)
        _, _, _, num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
        dets = np.concatenate(
            [
                final_boxes[:num[0]],
                np.array(final_scores)[:num[0]].reshape(-1, 1),
                np.array(final_cls_inds)[:num[0]].reshape(-1, 1)
            ], axis=-1)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            origin_img = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        return origin_img
    
    def get_fps(self):
        # warmup
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(20):
            _ = self._infer(img)
        t1 = time.perf_counter()
        _ = self._infer(img)
        self.node.loginfo(f"YOLOv7: {1/(time.perf_counter() - t1)} FPS")
    
    @staticmethod
    def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img


if __name__=="__main__":
    # to build tensorrt engine
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("--fp16", action='store_true', 
                        help="The precision mode to build in, either 'fp32', 'fp16' default: 'fp32'")
    opt = parser.parse_args()
    print(opt)

    # builds and saves tensorrt engine
    yolo = YOLOv7TRT(
        None, onnx_path=opt.onnx, fp16=opt.fp16)
    