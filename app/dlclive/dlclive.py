import os
import glob
import warnings
import numpy as np
import tensorflow as tf
import typing
import ruamel.yaml
from pathlib import Path
from typing import Optional, Tuple, List

try:
    TFVER = [int(v) for v in tf.__version__.split(".")]
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
except Exception:
    pass

from dlclive.graph import (
    read_graph,
    finalize_graph,
    get_output_nodes,
    extract_graph,
)
from dlclive.pose import extract_cnn_output, argmax_pose_predict, multi_pose_predict

from dlclive import utils
from dlclive.exceptions import DLCLiveError, DLCLiveWarning

if typing.TYPE_CHECKING:
    from dlclive.processor import Processor


class DLCLive(object):
    PARAMETERS = (
        "path",
        "cfg",
        "model_type",
        "precision",
        "cropping",
        "dynamic",
        "resize",
        "processor",
    )

    def __init__(
        self,
        model_path: str,
        model_type: str = "base",
        precision: str = "FP32",
        tf_config=None,
        cropping: Optional[List[int]] = None,
        dynamic: Tuple[bool, float, float] = (False, 0.5, 10),
        resize: Optional[float] = None,
        convert2rgb: bool = True,
        processor: Optional["Processor"] = None,
    ):
        self.path = model_path
        self.cfg = None  # type: typing.Optional[dict]
        self.model_type = model_type
        self.tf_config = tf_config
        self.precision = precision
        self.cropping = cropping
        self.dynamic = dynamic
        self.dynamic_cropping = None
        self.resize = resize
        self.processor = processor
        self.convert2rgb = convert2rgb
        self.sess = None
        self.inputs = None
        self.outputs = None
        self.tflite_interpreter = None
        self.pose = None
        self.is_initialized = False

        # checks

        if self.model_type == "tflite" and self.dynamic[0]:
            self.dynamic = (False, *self.dynamic[1:])
            warnings.warn(
                "Dynamic cropping is not supported for tensorflow lite inference. Dynamic cropping will not be used...",
                DLCLiveWarning,
            )

        self.read_config()

    def read_config(self):
        cfg_path = Path(self.path).resolve() / "pose_cfg.yaml"

        if not cfg_path.exists():
            raise FileNotFoundError(
                f"The pose configuration file for the exported model at {str(cfg_path)} was not found. Please check the path to the exported model directory"
            )

        ruamel_file = ruamel.yaml.YAML()
        self.cfg = ruamel_file.load(open(str(cfg_path), "r"))

    @property
    def parameterization(self) -> dict:
        return {param: getattr(self, param) for param in self.PARAMETERS}

    def process_frame(self, frame):
        if frame.dtype != np.uint8:
            frame = utils.convert_to_ubyte(frame)

        if self.cropping:
            frame = frame[
                self.cropping[2] : self.cropping[3], self.cropping[0] : self.cropping[1]
            ]

        if self.dynamic[0]:
            if self.pose is not None:
                detected = self.pose[:, 2] > self.dynamic[1]

                if np.any(detected):
                    x = self.pose[detected, 0]
                    y = self.pose[detected, 1]

                    x1 = int(max([0, int(np.amin(x)) - self.dynamic[2]]))
                    x2 = int(min([frame.shape[1], int(np.amax(x)) + self.dynamic[2]]))
                    y1 = int(max([0, int(np.amin(y)) - self.dynamic[2]]))
                    y2 = int(min([frame.shape[0], int(np.amax(y)) + self.dynamic[2]]))
                    self.dynamic_cropping = [x1, x2, y1, y2]

                    frame = frame[y1:y2, x1:x2]

                else:
                    self.dynamic_cropping = None

        if self.resize != 1:
            frame = utils.resize_frame(frame, self.resize)

        if self.convert2rgb:
            frame = utils.img_to_rgb(frame)

        return frame

    def init_inference(self, frame=None, **kwargs):
        # get model file

        model_file = glob.glob(os.path.normpath(self.path + "/*.pb"))[0]
        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                "The model file {} does not exist.".format(model_file)
            )

        # process frame

        if frame is None and (self.model_type == "tflite"):
            raise DLCLiveError(
                "No image was passed to initialize inference. An image must be passed to the init_inference method"
            )

        if frame is not None:
            if frame.ndim == 2:
                self.convert2rgb = True
            processed_frame = self.process_frame(frame)

        # load model

        if self.model_type == "base":
            graph_def = read_graph(model_file)
            graph = finalize_graph(graph_def)
            self.sess, self.inputs, self.outputs = extract_graph(
                graph, tf_config=self.tf_config
            )

        elif self.model_type == "tflite":
            ###
            # the frame size needed to initialize the tflite model as
            # tflite does not support saving a model with dynamic input size
            ###

            # get input and output tensor names from graph_def
            graph_def = read_graph(model_file)
            graph = finalize_graph(graph_def)
            output_nodes = get_output_nodes(graph)
            output_nodes = [on.replace("DLC/", "") for on in output_nodes]

            tf_version_2 = tf.__version__[0] == "2"

            if tf_version_2:
                converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
                    model_file,
                    ["Placeholder"],
                    output_nodes,
                    input_shapes={
                        "Placeholder": [
                            1,
                            processed_frame.shape[0],
                            processed_frame.shape[1],
                            3,
                        ]
                    },
                )
            else:
                converter = tf.lite.TFLiteConverter.from_frozen_graph(
                    model_file,
                    ["Placeholder"],
                    output_nodes,
                    input_shapes={
                        "Placeholder": [
                            1,
                            processed_frame.shape[0],
                            processed_frame.shape[1],
                            3,
                        ]
                    },
                )

            try:
                tflite_model = converter.convert()
            except Exception:
                raise DLCLiveError(
                    (
                        "This model cannot be converted to tensorflow lite format. "
                        "To use tensorflow lite for live inference, "
                        "make sure to set TFGPUinference=False "
                        "when exporting the model from DeepLabCut"
                    )
                )

            self.tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
            self.tflite_interpreter.allocate_tensors()
            self.inputs = self.tflite_interpreter.get_input_details()
            self.outputs = self.tflite_interpreter.get_output_details()

        else:
            raise DLCLiveError(
                "model_type = {} is not supported. model_type must be 'base', 'tflite', or 'tensorrt'".format(
                    self.model_type
                )
            )

        # get pose of first frame (first inference is often very slow)

        if frame is not None:
            pose = self.get_pose(frame, **kwargs)
        else:
            pose = None

        self.is_initialized = True

        return pose

    def get_pose(self, frame=None, **kwargs):
        if frame is None:
            raise DLCLiveError("No frame provided for live pose estimation")

        frame = self.process_frame(frame)

        if self.model_type == "base":
            pose_output = self.sess.run(
                self.outputs, feed_dict={self.inputs: np.expand_dims(frame, axis=0)}
            )

        elif self.model_type == "tflite":
            self.tflite_interpreter.set_tensor(
                self.inputs[0]["index"],
                np.expand_dims(frame, axis=0).astype(np.float32),
            )
            self.tflite_interpreter.invoke()

            if len(self.outputs) > 1:
                pose_output = [
                    self.tflite_interpreter.get_tensor(self.outputs[0]["index"]),
                    self.tflite_interpreter.get_tensor(self.outputs[1]["index"]),
                ]
            else:
                pose_output = self.tflite_interpreter.get_tensor(
                    self.outputs[0]["index"]
                )

        else:
            raise DLCLiveError(
                "model_type = {} is not supported. model_type must be 'base', 'tflite', or 'tensorrt'".format(
                    self.model_type
                )
            )

        if len(pose_output) > 1:
            scmap, locref = extract_cnn_output(pose_output, self.cfg)
            num_outputs = self.cfg.get("num_outputs", 1)
            if num_outputs > 1:
                self.pose = multi_pose_predict(
                    scmap, locref, self.cfg["stride"], num_outputs
                )
            else:
                self.pose = argmax_pose_predict(scmap, locref, self.cfg["stride"])
        else:
            pose = np.array(pose_output[0])
            self.pose = pose[:, [1, 0, 2]]

        if self.resize is not None:
            self.pose[:, :2] *= 1 / self.resize

        if self.cropping is not None:
            self.pose[:, 0] += self.cropping[0]
            self.pose[:, 1] += self.cropping[2]

        if self.dynamic_cropping is not None:
            self.pose[:, 0] += self.dynamic_cropping[0]
            self.pose[:, 1] += self.dynamic_cropping[2]

        # process the pose

        if self.processor:
            self.pose = self.processor.process(self.pose, **kwargs)

        return self.pose

    def close(self):
        self.sess.close()
        self.sess = None
        self.is_initialized = False
