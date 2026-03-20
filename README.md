# SkyFoodV2-Vision-OAK

Standalone perception pipeline for the **SkyFoodV2 delivery robot** using a **Luxonis OAK-D Pro PoE** camera. Runs on the host Jetson over Ethernet (PoE switch), publishes all vision data to ROS 2 Humble.

| Layer | Technology |
|---|---|
| **Camera** | Luxonis OAK-D Pro PoE (RVC4) |
| **Device API** | [DepthAI v3](https://docs.luxonis.com/software-v3/) |
| **NN Wrapper** | [depthai-nodes](https://github.com/luxonis/depthai-nodes) |
| **Robot OS** | ROS 2 Humble (on NVIDIA Jetson) |
| **Image transport** | `sensor_msgs/Image` via `cv_bridge` / numpy |
| **Detection msgs** | `vision_msgs/Detection3DArray` |
| **Language** | Python 3.10+ |
| **Connectivity** | Ethernet / PoE Switch |

## Hardware Requirements

- Luxonis **OAK-D Pro** (PoE version)
- PoE switch connecting the camera and the Jetson on the same subnet
- NVIDIA Jetson (Orin / Xavier) running **Ubuntu 22.04** + **ROS 2 Humble**

## Software Requirements

```bash
# Install ROS 2 Humble (follow official guide) and source it
source /opt/ros/humble/setup.bash
sudo apt install ros-humble-vision-msgs
```

> **Note:** `rclpy` is a system package installed with ROS 2 — do not install it via pip.

## IP Address Setup (PoE)

### Step 1 – Find the camera's IP address

The OAK-D Pro PoE assigns itself a **link-local** address (`169.254.x.x`) by default, or a DHCP address if a DHCP server is on the switch.

```bash
# Option A: automated discovery
python3 -c "import depthai as dai; [print(d) for d in dai.Device.getAllAvailableDevices()]"

# Option B: scan the link-local subnet (camera connected via PoE)
arp -a | grep 169.254
```

### Step 2 – Assign a static IP (recommended for production)

For reliable operation in the robot:

```bash
# On the Jetson – set a static IP on the PoE interface (e.g. eth1)
sudo ip addr add 169.254.1.1/16 dev eth1
sudo ip link set eth1 up
```

Configure the OAK-D Pro PoE static IP using the [Luxonis PoE Guide](https://docs.luxonis.com/en/latest/pages/deployment/poe_deployment/).  
The default link-local address is usually **`169.254.1.222`**.

### Step 3 – Edit config.py

Open [`config.py`](config.py) and set:

```python
OAK_IP = "169.254.1.222"   # ← change to your camera's actual IP
```

### Step 4 – Test connectivity

```bash
ping 169.254.1.222
# Expected: replies with ~1 ms latency
```

## Python Environment Setup

```bash
git clone https://github.com/MekhyW/SkyFoodV2-Vision-OAK.git
cd SkyFoodV2-Vision-OAK
pip install -r requirements.txt
```

## Training the Custom Dock Detector

The charging dock model requires custom training. Follow these steps:

#### 1. Dataset Collection

- Collect **300–500 images** of the charging dock in various:
  - Lighting conditions (office fluorescent, daylight, low-light)
  - Distances (0.3 m – 5 m)
  - Angles (frontal, slight left/right/up/down offset)
  - Backgrounds (varied floors, walls, clutter)

#### 2. Annotation

Use [Roboflow](https://roboflow.com):

1. Create a new project → **Object Detection**.
2. Upload images.
3. Draw bounding boxes around the dock, label as `charging_dock`.
4. Apply augmentations: flip, brightness, mosaic, crop.
5. Export as **YOLOv7 PyTorch** format.

#### 3. Model Training

Use a small YOLO variant (e.g. **YOLOv7-tiny** or **YOLOv6-nano**):

```bash
# Example with YOLOv6 (Meituan repo)
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt

python tools/train.py \
  --batch 32 \
  --conf configs/yolov6n.py \
  --data data/dock.yaml \
  --epochs 100 \
  --device 0
```

#### 4. Export to ONNX

```bash
python deploy/ONNX/export_onnx.py \
  --weights runs/train/exp/weights/best.pt \
  --img-size 320 320 \
  --batch 1
```

#### 5. Convert to .blob (Luxonis blob converter)

1. Go to [https://tools.luxonis.com/blob-converter/](https://tools.luxonis.com/blob-converter/).
2. Upload `best.onnx`.
3. Select **RVC4** platform.
4. Set input shape to `[1, 3, 320, 320]`.
5. Download the resulting `.blob`.
6. Place it at:

```
models/dock_detector.blob
```

#### 6. Update config

In `config.py`, update:
```python
DOCK_INPUT_WIDTH  = 320
DOCK_INPUT_HEIGHT = 320
```

## Running the Pipeline

```bash
# Source ROS 2 first
source /opt/ros/humble/setup.bash

# Full pipeline (all 4 phases) – requires all model blobs present
python3 main.py

# Specify a different camera IP
python3 main.py --ip 192.168.1.50

# Run only Phase 1 (color + depth, no NN)
python3 main.py --phase 1

# Run without ROS 2 (headless – for quick camera validation)
python3 main.py --no-ros --phase 1
```

## Published ROS 2 Topics

| Topic | Type | Phase | Hz |
|---|---|---|---|
| `/color/image_raw` | `sensor_msgs/Image` | 1 | 30 |
| `/stereo/depth` | `sensor_msgs/Image` | 1 | 30 |
| `/camera_info` | `sensor_msgs/CameraInfo` | 1 | 30 |
| `/imu` | `sensor_msgs/Imu` | 2 | 400 |
| `/perception/human_detections` | `vision_msgs/Detection3DArray` | 3 | ≤30 |
| `/perception/face_detections` | `vision_msgs/Detection3DArray` | 4 | ≤30 |
| `/perception/dock_detections` | `vision_msgs/Detection3DArray` | 4 | ≤30 |

## Hardware Verification

Perform these checks with the OAK-D Pro PoE physically connected on the PoE switch:

### Phase 1 — Color + Depth

```bash
source /opt/ros/humble/setup.bash
python3 main.py --phase 1 &

# In another terminal:
ros2 topic list           # must show /color/image_raw, /stereo/depth, /camera_info
ros2 topic hz /color/image_raw   # expect ~30 Hz
ros2 topic hz /stereo/depth      # expect ~30 Hz

# Visual check:
rviz2
# Add: Displays → Image → /color/image_raw
# Add: Displays → Image → /stereo/depth  (set min/max depth range)
```

✅ **Pass criteria:** Colour image streams at 30 Hz with no tearing; depth map shows warm-to-cold gradient with no all-black frames.

### Phase 2 — IMU

```bash
python3 main.py --phase 2 &
ros2 topic hz /imu                        # expect ~400 Hz
ros2 topic echo /imu --once               # linear_acceleration / angular_velocity non-zero
```

✅ **Pass criteria:** IMU data at ≥200 Hz; `linear_acceleration.z ≈ 9.81 m/s²` when camera is flat.

### Phase 3 — Human Detection

```bash
python3 main.py --phase 3 &
ros2 topic echo /perception/human_detections   # walk in front of camera
```

✅ **Pass criteria:** Detections appear within 1 s of a person entering the frame, with `bbox.center.position.z > 0`.

### Phase 4 — Face + Dock Detection

```bash
python3 main.py &
ros2 topic echo /perception/face_detections    # face in front of camera
ros2 topic echo /perception/dock_detections    # charging dock in camera view
```

✅ **Pass criteria:** Face detected at ≥ 2 m distance; dock detected at ≥ 1 m with `confidence > 0.6`.

## Software Verification (no hardware)

Run these checks without a camera to validate code correctness:

```bash
# Syntax / import check
python3 -m py_compile main.py ros2_publisher.py config.py
echo "Syntax OK"

# Verify DepthAI install
python3 -c "import depthai; print('depthai', depthai.__version__)"

# Verify depthai-nodes install
python3 -c "from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork; print('depthai-nodes OK')"

# Check available DepthAI devices (no error = library working)
python3 -c "import depthai as dai; print('Devices:', dai.Device.getAllAvailableDevices())"
```
