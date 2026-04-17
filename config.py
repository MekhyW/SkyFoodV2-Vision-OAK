# ── PoE Device ────────────────────────────────────────────────────────────────
# IP address of the OAK-D Pro PoE camera on the network.
# If set to None, the first USB/PoE device found will be used (useful for dev).
OAK_IP = "169.254.1.222"

# ── Camera ────────────────────────────────────────────────────────────────────
COLOR_FPS = 30           # Color camera frame rate (fps)
MONO_RESOLUTION = "400p" # Mono camera resolution: "400p" or "720p"
COLOR_RESOLUTION = "1080p"

# ── Stereo Depth ──────────────────────────────────────────────────────────────
DEPTH_PRESET = "HIGH_ACCURACY"  # "HIGH_ACCURACY" or "HIGH_DENSITY"
DEPTH_ALIGN_TO_COLOR = True     # Align depth map to color camera frame

# ── IMU ───────────────────────────────────────────────────────────────────────
IMU_RATE_HZ = 400    # Batch report frequency for accelerometer + gyroscope
IMU_BATCH_SIZE = 1   # How many IMU packets to batch before sending to host

# ── Human Detection ─────────────────────────────────────────────────
HUMAN_MODEL_PATH   = "models/yolov6n-r2-288x512"
HUMAN_CONFIDENCE   = 0.5       # Detection threshold
HUMAN_IOU_THRESHOLD = 0.4
HUMAN_NUM_CLASSES  = 80        # COCO classes
# COCO class index for "person" is 0
HUMAN_LABEL_INDEX  = 0

# YOLOv6-nano I/O dimensions (must match blob)
HUMAN_INPUT_WIDTH  = 512
HUMAN_INPUT_HEIGHT = 288

# ── Face Detection ──────────────────────────────────────────────────
FACE_MODEL_PATH    = "models/yunet-s-240x320"
FACE_CONFIDENCE    = 0.5
FACE_INPUT_WIDTH   = 320
FACE_INPUT_HEIGHT  = 240

# ── ROS 2 Topics ──────────────────────────────────────────────────────────────
TOPIC_COLOR        = "/color/image_raw"
TOPIC_DEPTH        = "/stereo/depth"
TOPIC_CAMERA_INFO  = "/camera_info"
TOPIC_IMU          = "/imu"
TOPIC_HUMANS       = "/perception/human_detections"
TOPIC_FACES        = "/perception/face_detections"

# Frame IDs expected by RViz2 / Nav2
FRAME_COLOR        = "oak_rgb_camera_optical_frame"
FRAME_DEPTH        = "oak_right_camera_optical_frame"
FRAME_IMU          = "oak_imu_frame"
