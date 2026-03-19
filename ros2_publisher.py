"""
Receives DepthAI V3 output queues from main.py and publishes all perception
data onto standard ROS 2 topics.

Topics published:
  /color/image_raw           sensor_msgs/Image
  /stereo/depth              sensor_msgs/Image
  /camera_info               sensor_msgs/CameraInfo
  /imu                       sensor_msgs/Imu
  /perception/human_detections  vision_msgs/Detection3DArray
  /perception/face_detections   vision_msgs/Detection3DArray
  /perception/dock_detections   vision_msgs/Detection3DArray

Prerequisites on the Jetson:
  source /opt/ros/humble/setup.bash
  sudo apt install ros-humble-vision-msgs
"""

from __future__ import annotations
import sys
from typing import Optional
import numpy as np
import depthai as dai # DepthAI V3
import config
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from std_msgs.msg import Header
    from sensor_msgs.msg import Image, CameraInfo, Imu
    try:
        from vision_msgs.msg import Detection3D, Detection3DArray, BoundingBox3D, ObjectHypothesisWithPose
        _VISION_MSGS = True
    except ImportError:
        print("[ros2_publisher] WARNING: vision_msgs not found. Install with: sudo apt install ros-humble-vision-msgs\nDetection topics will be disabled.")
        _VISION_MSGS = False
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    print("[ros2_publisher] ERROR: rclpy not found. Source ROS 2 before running: source /opt/ros/humble/setup.bash")
    sys.exit(1)


# ── Conversion helpers ────────────────────────────────────────────────────────

def _make_header(stamp: Optional[rclpy.time.Time] = None, frame_id: str = "") -> Header:
    h = Header()
    h.stamp = stamp or rclpy.time.Time().to_msg()
    h.frame_id = frame_id
    return h


def _dai_image_to_ros(frame: dai.ImgFrame, frame_id: str, node: Node) -> Image:
    """Convert a dai.ImgFrame to sensor_msgs/Image."""
    msg = Image()
    msg.header = _make_header(node.get_clock().now().to_msg(), frame_id)
    cv_frame = frame.getCvFrame()
    msg.height, msg.width = cv_frame.shape[:2]
    msg.step = cv_frame.strides[0]
    if cv_frame.ndim == 2:
        # Depth (uint16) or grayscale
        if cv_frame.dtype == np.uint16:
            msg.encoding = "16UC1"
        else:
            msg.encoding = "mono8"
    else:
        msg.encoding = "bgr8"
    msg.data = cv_frame.tobytes()
    msg.is_bigendian = False
    return msg


def _make_camera_info(frame: dai.ImgFrame, frame_id: str, node: Node) -> CameraInfo:
    """Build a minimal CameraInfo message (identity model – calibrate properly)."""
    msg = CameraInfo()
    msg.header = _make_header(node.get_clock().now().to_msg(), frame_id)
    h, w = frame.getCvFrame().shape[:2]
    msg.height = h
    msg.width = w
    # Placeholder intrinsics – replace with calibration data from the device
    fx = fy = float(w)   # rough approximation
    cx, cy = w / 2.0, h / 2.0
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return msg


def _dai_imu_to_ros(imu_data: dai.IMUData, node: Node) -> list[Imu]:
    """Convert dai.IMUData packet batches into a list of sensor_msgs/Imu."""
    messages = []
    packets = imu_data.packets
    for pkt in packets:
        msg = Imu()
        msg.header = _make_header(node.get_clock().now().to_msg(), config.FRAME_IMU)
        accel = pkt.acceleroMeter # Linear acceleration (m/s²)
        msg.linear_acceleration.x = accel.x
        msg.linear_acceleration.y = accel.y
        msg.linear_acceleration.z = accel.z
        msg.linear_acceleration_covariance[0] = -1  # unknown
        gyro = pkt.gyroscope # Angular velocity (rad/s)
        msg.angular_velocity.x = gyro.x
        msg.angular_velocity.y = gyro.y
        msg.angular_velocity.z = gyro.z
        msg.angular_velocity_covariance[0] = -1  # unknown
        if hasattr(pkt, 'rotationVector'): # Orientation (quaternion from rotation vector)
            rv = pkt.rotationVector
            msg.orientation.w = rv.real
            msg.orientation.x = rv.i
            msg.orientation.y = rv.j
            msg.orientation.z = rv.k
        else:
            msg.orientation_covariance[0] = -1  # orientation not available
        messages.append(msg)
    return messages


def _spatial_detections_to_ros(detections: dai.SpatialImgDetections, label: str, frame_id: str, node: Node) -> "Detection3DArray":
    """Convert SpatialImgDetections → vision_msgs/Detection3DArray."""
    arr = Detection3DArray()
    arr.header = _make_header(node.get_clock().now().to_msg(), frame_id)
    for det in detections.detections:
        d3d = Detection3D()
        d3d.header = arr.header
        hyp = ObjectHypothesisWithPose() # Hypothesis
        hyp.hypothesis.class_id = label
        hyp.hypothesis.score = det.confidence
        d3d.results.append(hyp)
        sp = det.spatialCoordinates # 3-D bounding box centre from spatial coordinates (mm → m)
        bbox = BoundingBox3D()
        bbox.center.position.x = sp.x / 1000.0
        bbox.center.position.y = sp.y / 1000.0
        bbox.center.position.z = sp.z / 1000.0
        bbox.size.x = abs(det.xmax - det.xmin) * sp.z / 1000.0
        bbox.size.y = abs(det.ymax - det.ymin) * sp.z / 1000.0
        bbox.size.z = 0.3  # depth unknown, use a sensible default (m)
        d3d.bbox = bbox
        arr.detections.append(d3d)
    return arr


def _generic_detections_to_ros(nn_data: dai.NNData, locations: dai.SpatialLocations, label: str, frame_id: str, confidence_threshold: float, input_w: int, input_h: int, node: Node) -> "Detection3DArray":
    """
    Convert generic NNData + SpatialLocations into a Detection3DArray.
    Expects NN output layer as flat fp16 array: [x_min, y_min, x_max, y_max, conf, cls, ...]
    """
    arr = Detection3DArray()
    arr.header = _make_header(node.get_clock().now().to_msg(), frame_id)
    try:
        layer = nn_data.getFirstLayerFp16()
    except Exception:
        return arr
    n = len(layer) // 6
    loc_list = locations.spatialLocations if locations else []
    for i in range(min(n, len(loc_list))):
        base = i * 6
        conf = layer[base + 4]
        if conf < confidence_threshold:
            continue
        loc = loc_list[i]
        d3d = Detection3D()
        d3d.header = arr.header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = label
        hyp.hypothesis.score = float(conf)
        d3d.results.append(hyp)
        bbox = BoundingBox3D()
        bbox.center.position.x = loc.spatialCoordinates.x / 1000.0
        bbox.center.position.y = loc.spatialCoordinates.y / 1000.0
        bbox.center.position.z = loc.spatialCoordinates.z / 1000.0
        bbox.size.x = abs(layer[base + 2] - layer[base + 0]) * loc.spatialCoordinates.z / 1000.0
        bbox.size.y = abs(layer[base + 3] - layer[base + 1]) * loc.spatialCoordinates.z / 1000.0
        bbox.size.z = 0.2
        d3d.bbox = bbox
        arr.detections.append(d3d)
    return arr


# ── Publisher node ────────────────────────────────────────────────────────────

class PerceptionPublisher:
    """
    Wraps a rclpy Node and pumps data from DepthAI output queues into ROS 2.
    """
    QUEUE_TIMEOUT_MS = 5   # non-blocking queue poll timeout
    def __init__(self, device: dai.Device, queues: dict[str, str]) -> None:
        rclpy.init()
        self._node = Node("skyfood_perception")
        self._device = device
        self._q_names = queues
        self._qs: dict[str, dai.DataOutputQueue] = {} # Get output queue handles
        for logical_name, stream_name in queues.items():
            self._qs[logical_name] = device.getOutputQueue(name=stream_name, maxSize=4, blocking=False)
        self._pub_color = self._node.create_publisher(Image, config.TOPIC_COLOR, 10)
        self._pub_depth = self._node.create_publisher(Image, config.TOPIC_DEPTH, 10)
        self._pub_info  = self._node.create_publisher(CameraInfo, config.TOPIC_CAMERA_INFO, 10)
        self._pub_imu   = self._node.create_publisher(Imu, config.TOPIC_IMU, 100)
        if _VISION_MSGS:
            self._pub_humans = self._node.create_publisher(Detection3DArray, config.TOPIC_HUMANS, 10)
            self._pub_faces  = self._node.create_publisher(Detection3DArray, config.TOPIC_FACES, 10)
            self._pub_dock   = self._node.create_publisher(Detection3DArray, config.TOPIC_DOCK, 10)
        else:
            self._pub_humans = self._pub_faces = self._pub_dock = None
        self._node.get_logger().info("PerceptionPublisher initialised.")

    def _get(self, name: str):
        if name not in self._qs:
            return None
        return self._qs[name].tryGet()

    def spin(self, pipeline: dai.Pipeline) -> None:
        """Run the publisher loop until the pipeline stops or Ctrl-C."""
        log = self._node.get_logger()
        log.info("Starting perception publisher loop …")
        while pipeline.isRunning():
            rclpy.spin_once(self._node, timeout_sec=0)
            self._process_color()
            self._process_imu()
            self._process_humans()
            self._process_faces()
            self._process_dock()

    def _process_color(self) -> None:
        frame: Optional[dai.ImgFrame] = self._get("color")
        if frame is None:
            return
        img_msg = _dai_image_to_ros(frame, config.FRAME_COLOR, self._node)
        self._pub_color.publish(img_msg)
        depth_frame: Optional[dai.ImgFrame] = self._get("depth")
        if depth_frame is not None:
            depth_msg = _dai_image_to_ros(depth_frame, config.FRAME_DEPTH, self._node)
            self._pub_depth.publish(depth_msg)
        info_msg = _make_camera_info(frame, config.FRAME_COLOR, self._node)
        self._pub_info.publish(info_msg)

    def _process_imu(self) -> None:
        imu_data: Optional[dai.IMUData] = self._get("imu")
        if imu_data is None:
            return
        for imu_msg in _dai_imu_to_ros(imu_data, self._node):
            self._pub_imu.publish(imu_msg)

    def _process_humans(self) -> None:
        if not _VISION_MSGS or "humans" not in self._qs:
            return
        detections: Optional[dai.SpatialImgDetections] = self._get("humans")
        if detections is None:
            return
        msg = _spatial_detections_to_ros(detections, "person", config.FRAME_COLOR, self._node)
        self._pub_humans.publish(msg)

    def _process_faces(self) -> None:
        if not _VISION_MSGS or "faces_det" not in self._qs:
            return
        nn_data: Optional[dai.NNData] = self._get("faces_det")
        locations = self._get("faces_loc")
        if nn_data is None or locations is None:
            return
        msg = _generic_detections_to_ros(nn_data, locations, "face", config.FRAME_COLOR, config.FACE_CONFIDENCE, config.FACE_INPUT_WIDTH, config.FACE_INPUT_HEIGHT, self._node)
        self._pub_faces.publish(msg)

    def _process_dock(self) -> None:
        if not _VISION_MSGS or "dock_det" not in self._qs:
            return
        nn_data: Optional[dai.NNData] = self._get("dock_det")
        locations = self._get("dock_loc")
        if nn_data is None or locations is None:
            return
        msg = _generic_detections_to_ros(nn_data, locations, "charging_dock", config.FRAME_COLOR, config.DOCK_CONFIDENCE, config.DOCK_INPUT_WIDTH, config.DOCK_INPUT_HEIGHT, self._node)
        self._pub_dock.publish(msg)

    def destroy(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()
        print("[ros2_publisher] Node destroyed.")
