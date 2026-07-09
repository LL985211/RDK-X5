# Adaptive Polarization Imaging and Visual Tracking System on RDK X5

This project implements an embedded vision system for glare-heavy scenes such as water surfaces, wet roads, reflective metal, glass, and UAV inspection targets. It combines an RDK X5 edge AI board, an STM32-controlled rotating linear polarizer, camera capture, image quality scoring, YOLOv8n BPU inference, DaSiamRPN single-object tracking, local recording, and optional RTSP streaming.

The current folder contains:

| File | Description |
| --- | --- |
| `RDKX5pzg21-high MJPG-T-C-RPN.py` | Main Python application for RDK X5, using YOLOv8n BPU detection and DaSiamRPN tracking. |
| `2026嵌入式大赛应用赛道作品.docx` | Competition report describing the system design, implementation, and results. |
| `13ca1b07ab06b731c25ccccd37dcd7a2.mp4` | Demonstration video, MP4 container with AVC/H.264 video encoding. |

## Features

- 0 to 180 degree automatic polarization scan with 10 degree sampling.
- Five-factor image quality evaluation: brightness, contrast, color saturation, detail, and glare suppression.
- Best-angle selection and automatic return to the highest-scoring polarizer angle.
- Camera preview with OpenCV windows for live view and best-frame view.
- YOLOv8n object detection accelerated by the RDK X5 BPU.
- YOLO-assisted target selection followed by DaSiamRPN single-object tracking.
- Tracking video recording with target bounding boxes.
- CSV and JSON evaluation output for each capture round.
- Optional RTSP streaming for live video and best-frame preview when MediaMTX is running.

## System Architecture

The system uses a split embedded architecture:

- **RDK X5** handles camera acquisition, image quality evaluation, BPU inference, DaSiamRPN tracking, UI display, video recording, data export, and RTSP push.
- **STM32** handles real-time polarizer motor execution, angle feedback, and low-level actuator protection.
- **Camera and polarizer module** provide the optical front end.
- **Optional gimbal or flight controller** can consume tracking and visual servo information in an extended deployment.

The main runtime loop has three user-facing modes:

1. `SCAN`: run polarization scanning and best-frame selection.
2. `DETECT`: run continuous YOLO detection and show selectable target boxes.
3. `TRACKING`: track the selected target with DaSiamRPN and optionally record tracking video.

## Hardware Requirements

- D-Robotics / Horizon RDK X5 board.
- USB or MIPI camera supported by OpenCV.
- STM32 controller connected through serial.
- Motor driver and gear-driven rotating linear polarizer.
- Optional MediaMTX RTSP server.
- Optional gimbal or flight controller interface.

The script currently searches these serial ports:

```text
/dev/ttyUSB0
/dev/ttyACM0
/dev/ttyAMA0
/dev/ttyS0
/dev/ttyTHS1
```

Serial settings are `115200 bps`; commands are sent as text lines such as `T0`, `T180`, and `T<best_angle>`.

## Software Requirements

The application is intended to run on the RDK X5 Linux environment.

Required runtime components:

- Python 3
- OpenCV (`cv2`)
- NumPy
- PySerial
- Horizon `hobot_dnn`
- `rdk_model_zoo`
- `ultralytics_yolo_det`
- OpenCV tracking APIs, including DaSiamRPN and CSRT fallback support
- FFmpeg, for RTSP push and video handling
- MediaMTX, optional, for RTSP serving

The script expects the BPU model at:

```text
~/rdk_model_zoo/samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin
```

It also appends these paths at startup:

```text
/home/sunrise/rdk_model_zoo
/home/sunrise/rdk_model_zoo/samples/vision/ultralytics_yolo/runtime/python
```

Adjust these paths in the script if your RDK X5 user name or model location is different.

## Running

On the RDK X5, run:

```bash
python3 "RDKX5pzg21-high MJPG-T-C-RPN.py"
```

Startup checks include:

- BPU model file existence.
- Camera initialization.
- Serial connection to STM32.
- Optional RTSP server availability on port `8554`.

## Controls

| Mode | Action |
| --- | --- |
| `SCAN` | Press `A` to start one polarization scan round. |
| `SCAN` | Press `T` to enter continuous target detection mode. |
| `DETECT` | Click a bounding box to initialize the tracker and enter tracking mode. |
| `DETECT` | Press `X` to cancel detection and return to scan mode. |
| `TRACKING` | Press `A` or `X` to stop tracking and return to scan mode. |
| `TRACKING` | Press `C` to start or stop tracking video recording. |
| Any mode | Press `ESC` to exit. |

## Output Files

Capture rounds are saved under:

```text
~/Pictures/gear_capture/Capture_001
~/Pictures/gear_capture/Capture_002
...
```

Each round may contain:

- `capture_XXX.avi`: full capture video.
- `angle_000.jpg`, `angle_010.jpg`, ..., `angle_180.jpg`: sampled angle frames.
- `best_<angle>deg_<score>.jpg`: best-scoring frame.
- `evaluation_results.json`: structured evaluation results.
- `evaluation_scores.csv`: score table for each angle.

Tracking clips are saved under:

```text
~/Pictures/T/T1.avi
~/Pictures/T/T2.avi
...
```

When MediaMTX is running, RTSP streams are pushed to:

```text
rtsp://localhost:8554/live_video
rtsp://localhost:8554/best_frame
```

## Image Quality Score

The comprehensive score is calculated from five weighted terms:

```text
score = 0.40 * brightness
      + 0.25 * contrast
      + 0.15 * color
      + 0.12 * detail
      + 0.08 * glare_control
```

This lets the system choose a polarizer angle that balances exposure, texture, color, and glare suppression instead of relying on brightness alone.

## Reported Results

The competition report in this folder describes a complete RDK X5 and STM32 closed-loop prototype. Reported scan examples include:

- `Capture_001`: best angle 40 degrees, score 91.23.
- `Capture_002`: best angle 10 degrees, score 82.54.
- `Capture_003`: best angle 150 degrees, score 83.27.

The report also states that adaptive polarization improved tracking IoU from 48.5% with normal RGB input to 67.5% in the tested reflective scene, with the current report describing the tracking module as SiamRPN-based.

## Notes

- This script is hardware-dependent and is not expected to run correctly on a normal Windows or desktop Python environment without the RDK X5 BPU stack and connected devices.
- The current implementation uses OpenCV GUI windows, so a local display or X forwarding setup is required.
- If MediaMTX is not detected on port `8554`, RTSP streaming is disabled automatically and the rest of the application can still run.
- The model path, serial ports, image size, recording format, and scoring weights are defined near the top of the Python file.
