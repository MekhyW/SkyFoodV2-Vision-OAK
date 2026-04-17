"""
Runs on the Jetson (or any host connected to the OAK-D Pro PoE camera via
the PoE switch). Builds the full on-device pipeline in four phases, then
hands output queues to the ROS 2 publisher node.

Phase 1: Color + Depth
Phase 2: IMU
Phase 3: Human (person) spatial detection
Phase 4: Face detection
"""

import argparse
import os
import time
import depthai as dai
import config
from ros2_publisher import PerceptionPublisher


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SkyFoodV2-Vision-OAK Perception Pipeline")
    parser.add_argument("--ip", default=config.OAK_IP, help="IP address of the OAK-D Pro PoE camera. Set to empty string to use the first discovered device.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=4, help="Maximum pipeline phase to activate (default: 4 = full system).")
    parser.add_argument("--no-ros", action="store_true", help="Disable ROS 2 publishing (useful for standalone camera testing).")
    return parser.parse_args()


# ── Device helpers ────────────────────────────────────────────────────────────

def get_device(ip: str) -> dai.Device:
    """Return a dai.Device connected to the specified PoE IP (or any device)."""
    if ip:
        print(f"[main] Connecting to OAK-D Pro PoE at {ip} …")
        device_info = dai.DeviceInfo(ip)
        return dai.Device(device_info)
    else:
        print("[main] No IP specified – using first discovered device …")
        return dai.Device()


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(device: dai.Device, phase: int) -> tuple[dai.Pipeline, dict]:
    """
    Build the DepthAI V3 pipeline up to the requested phase.

    Returns
    -------
    pipeline : dai.Pipeline
    queues   : dict  – mapping of logical name → queue stream name (str)
                        used by the publisher to call device.getOutputQueue()
    """
    pipeline = dai.Pipeline(device)
    queues: dict[str, str] = {}

    print("[main] Phase 1: building color + stereo depth nodes …")
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(config.COLOR_FPS)
    cam_left = pipeline.create(dai.node.Camera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY if config.DEPTH_PRESET == "HIGH_ACCURACY" else dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    if config.DEPTH_ALIGN_TO_COLOR:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    cam_left.raw.link(stereo.left)
    cam_right.raw.link(stereo.right)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("color")
    cam_rgb.video.link(xout_rgb.input)
    queues["color"] = "color"
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    queues["depth"] = "depth"
    if phase < 2:
        return pipeline, queues

    print("[main] Phase 2: adding IMU node …")
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW, dai.IMUSensor.ROTATION_VECTOR], config.IMU_RATE_HZ)
    imu.setBatchReportThreshold(config.IMU_BATCH_SIZE)
    imu.setMaxBatchReports(10)
    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)
    queues["imu"] = "imu"
    if phase < 3:
        return pipeline, queues

    print("[main] Phase 3: adding human spatial detection …")
    if not os.path.isfile(config.HUMAN_MODEL_PATH):
        print(f"[main] WARNING: human model blob not found at '{config.HUMAN_MODEL_PATH}'. Skipping Phase 3. Please download the blob and restart.")
    else:
        manip_human = pipeline.create(dai.node.ImageManip) # ImageManip: resize color to NN input dimensions
        manip_human.initialConfig.setResize(config.HUMAN_INPUT_WIDTH, config.HUMAN_INPUT_HEIGHT)
        manip_human.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip_human.setKeepAspectRatio(False)
        cam_rgb.video.link(manip_human.inputImage)
        nn_humans = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        archive = dai.NNArchive(config.HUMAN_MODEL_PATH)
        nn_humans.setNNArchive(archive)
        nn_humans.setConfidenceThreshold(config.HUMAN_CONFIDENCE)
        nn_humans.setNumClasses(config.HUMAN_NUM_CLASSES)
        nn_humans.setCoordinateSize(4)
        nn_humans.setIouThreshold(config.HUMAN_IOU_THRESHOLD)
        nn_humans.setDepthLowerThreshold(100)    # mm
        nn_humans.setDepthUpperThreshold(15000)  # mm (15 m)
        nn_humans.setBoundingBoxScaleFactor(0.5)
        manip_human.out.link(nn_humans.input)
        stereo.depth.link(nn_humans.inputDepth)
        xout_humans = pipeline.create(dai.node.XLinkOut)
        xout_humans.setStreamName("humans")
        nn_humans.out.link(xout_humans.input)
        queues["humans"] = "humans"
    if phase < 4:
        return pipeline, queues

    print("[main] Phase 4: adding face detection …")
    if not os.path.isfile(config.FACE_MODEL_PATH):
        print(f"[main] WARNING: face model blob not found at '{config.FACE_MODEL_PATH}'. Skipping face detection. Please download the blob and restart.")
    else:
        manip_face = pipeline.create(dai.node.ImageManip)
        manip_face.initialConfig.setResize(config.FACE_INPUT_WIDTH, config.FACE_INPUT_HEIGHT)
        manip_face.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip_face.setKeepAspectRatio(False)
        cam_rgb.video.link(manip_face.inputImage)
        nn_faces = pipeline.create(dai.node.NeuralNetwork)
        archive = dai.NNArchive(config.FACE_MODEL_PATH)
        nn_faces.setNNArchive(archive)
        manip_face.out.link(nn_faces.input)
        slc_faces = pipeline.create(dai.node.SpatialLocationCalculator)
        slc_faces.setWaitForConfigInput(False)
        stereo.depth.link(slc_faces.inputDepth)
        script_faces = pipeline.create(dai.node.Script) # Script node: convert face NN output → SLC config
        script_faces.setScript(_make_slc_script(input_width=config.FACE_INPUT_WIDTH, input_height=config.FACE_INPUT_HEIGHT))
        nn_faces.out.link(script_faces.inputs["nn_out"])
        script_faces.outputs["slc_cfg"].link(slc_faces.inputConfig)
        xout_faces_det = pipeline.create(dai.node.XLinkOut)
        xout_faces_det.setStreamName("faces_det")
        nn_faces.out.link(xout_faces_det.input)
        queues["faces_det"] = "faces_det"
        xout_faces_loc = pipeline.create(dai.node.XLinkOut)
        xout_faces_loc.setStreamName("faces_loc")
        slc_faces.out.link(xout_faces_loc.input)
        queues["faces_loc"] = "faces_loc"

    return pipeline, queues


# ── Script template for SpatialLocationCalculator ────────────────────────────

def _make_slc_script(input_width: int, input_height: int) -> str:
    """
    Generate an on-device Script that converts generic NN bounding-box output
    into SpatialLocationCalculatorConfig messages.

    The script expects the NN output layer 'output' to contain detections in
    the format [batch, N, 6] where each detection is:
        [x_min_rel, y_min_rel, x_max_rel, y_max_rel, confidence, class_id]
    (normalised 0-1 relative to the NN input size).
    """
    return f"""
import Marshal

CONF_THRESHOLD = 0.5

while True:
    nn_data = node.inputs['nn_out'].get()
    layer = nn_data.getFirstLayerFp16()

    cfg = SpatialLocationCalculatorConfig()
    roi_list = []

    n_detections = len(layer) // 6
    for i in range(n_detections):
        base = i * 6
        x_min = layer[base + 0]
        y_min = layer[base + 1]
        x_max = layer[base + 2]
        y_max = layer[base + 3]
        conf  = layer[base + 4]

        if conf < CONF_THRESHOLD:
            continue

        roi_data = SpatialLocationCalculatorConfigData()
        roi_data.roi = Rect(Point2f(x_min, y_min), Point2f(x_max, y_max))
        roi_data.calculationAlgorithm = SpatialLocationCalculatorAlgorithm.MEDIAN
        roi_list.append(roi_data)

    cfg.setROIs(roi_list)
    node.outputs['slc_cfg'].send(cfg)
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = get_device(args.ip)
    print(f"[main] Device info: {device.getMxId()}")
    pipeline, queues = build_pipeline(device, args.phase)
    print(f"[main] Pipeline built. Active queues: {list(queues.keys())}")
    pipeline.start()
    print("[main] Pipeline started.")

    if args.no_ros: # Headless loop (no ROS) – useful for raw camera testing
        print("[main] ROS 2 disabled. Running headless loop. Press Ctrl-C to stop.")
        try:
            while pipeline.isRunning():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.stop()
            print("[main] Pipeline stopped.")
        return

    publisher = PerceptionPublisher(device=device, queues=queues)
    print("[main] Starting ROS 2 publisher loop …")
    try:
        publisher.spin(pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        publisher.destroy()
        print("[main] Shutdown complete.")


if __name__ == "__main__":
    main()