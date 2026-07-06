# 基于地瓜派 RDK X5 的自适应偏振成像与视觉追踪系统

本项目面向水面、湿滑路面、玻璃、金属、车窗等强反光场景，构建了一套嵌入式自适应偏振成像与视觉追踪系统。系统以地瓜派 RDK X5 作为视觉计算与任务调度主控，STM32 作为偏振机构实时控制器，结合相机采集、偏振角寻优、图像质量评价、YOLOv8n BPU 推理、ByteTrack 多目标跟踪、视频录制和可选 RTSP 推流，实现“光学抑光、图像评价、目标检测、目标跟踪、结果归档”的闭环流程。

当前目录包含：

| 文件 | 说明 |
| --- | --- |
| `RDKX5pzg21-high MJPG-T-C-ByteTrack.py` | RDK X5 端主程序，负责采集、评分、检测、跟踪、录制和推流。 |
| `2026嵌入式大赛应用赛道作品.docx最新.docx` | 参赛作品文档，包含系统设计、硬件组成、软件流程和实验结果。 |
| `lv_0_20260706165707.mp4` | 项目演示视频。 |

## 项目功能

- 支持 0° 到 180° 偏振角自动扫描，默认每 10° 采样一次。
- 从亮度、对比度、色彩饱和度、细节表现、眩光抑制五个维度评价图像质量。
- 自动选择综合得分最高的偏振角，并控制机构返回最佳角度。
- 使用 OpenCV 显示实时画面和最佳帧画面。
- 基于 RDK X5 BPU 运行 YOLOv8n 目标检测。
- 集成 ByteTrack 多目标跟踪，支持鼠标点击目标框选择跟踪 ID。
- 跟踪模式下可录制带目标框和 ID 的视频。
- 每轮采集自动保存关键帧、最佳帧、CSV 分数表和 JSON 结果。
- 检测到 MediaMTX 后，可通过 RTSP 输出实时画面和最佳帧画面。

## 系统组成

系统采用 RDK X5 与 STM32 分工协作的嵌入式架构：

- **RDK X5**：负责相机采集、图像质量评价、BPU 推理、ByteTrack 跟踪、界面显示、视频保存、数据导出和 RTSP 推流。
- **STM32**：负责偏振镜电机控制、角度执行、角度反馈和底层保护。
- **相机与线偏振镜机构**：构成可调偏振成像前端。
- **云台或飞控接口**：可在后续扩展中接收目标偏差和姿态控制量。

程序运行时包含三个主要模式：

1. `SCAN`：偏振扫描与最佳帧选择。
2. `DETECT`：持续检测并显示可选择的目标框。
3. `TRACKING`：跟踪选中的目标 ID，并可录制跟踪视频。

## 硬件要求

- 地瓜机器人 RDK X5 开发板。
- OpenCV 可访问的 USB 或 MIPI 摄像头。
- 通过串口连接的 STM32 控制板。
- 电机驱动板、齿轮传动机构和可旋转线偏振镜。
- 可选：MediaMTX RTSP 服务。
- 可选：云台或飞控执行接口。

脚本会依次尝试以下串口：

```text
/dev/ttyUSB0
/dev/ttyACM0
/dev/ttyAMA0
/dev/ttyS0
/dev/ttyTHS1
```

串口波特率为 `115200 bps`。当前程序发送的角度命令为文本行，例如 `T0`、`T180`、`T<best_angle>`，并从串口读取 `ANG:<angle>` 形式的角度反馈。

## 软件依赖

程序主要面向 RDK X5 Linux 环境运行。

运行依赖包括：

- Python 3
- OpenCV (`cv2`)
- NumPy
- PySerial
- Horizon `hobot_dnn`
- `rdk_model_zoo`
- `ultralytics_yolo_det`
- `cjm_byte_track`
- FFmpeg，用于视频处理和 RTSP 推流
- MediaMTX，可选，用于 RTSP 服务

BPU 模型默认路径为：

```text
~/rdk_model_zoo/samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin
```

脚本启动时会加入以下模块搜索路径：

```text
/home/sunrise/rdk_model_zoo
/home/sunrise/rdk_model_zoo/samples/vision/ultralytics_yolo/runtime/python
```

如果 RDK X5 的用户名、模型目录或 `rdk_model_zoo` 位置不同，需要在脚本顶部修改对应路径。

## 运行方法

在 RDK X5 上进入项目目录后执行：

```bash
python3 "RDKX5pzg21-high MJPG-T-C-ByteTrack.py"
```

启动阶段会检查：

- BPU 模型文件是否存在。
- 摄像头是否可以打开。
- STM32 串口是否连接成功。
- 本机 `8554` 端口是否存在 MediaMTX RTSP 服务。

## 操作按键

| 模式 | 操作 |
| --- | --- |
| `SCAN` | 按 `A` 启动一轮偏振扫描。 |
| `SCAN` | 按 `T` 进入持续目标检测模式。 |
| `DETECT` | 鼠标点击任意检测框，选择目标 ID 并进入跟踪模式。 |
| `DETECT` | 按 `X` 取消检测并返回扫描模式。 |
| `TRACKING` | 按 `A` 或 `X` 停止跟踪并返回扫描模式。 |
| `TRACKING` | 按 `C` 开始或停止录制跟踪视频。 |
| 任意模式 | 按 `ESC` 退出程序。 |

## 输出文件

偏振扫描结果默认保存到：

```text
~/Pictures/gear_capture/Capture_001
~/Pictures/gear_capture/Capture_002
...
```

每轮目录中可能包含：

- `capture_XXX.avi`：本轮完整采集视频。
- `angle_000.jpg`、`angle_010.jpg`、...、`angle_180.jpg`：各偏振角关键帧。
- `best_<angle>deg_<score>.jpg`：本轮最佳帧。
- `evaluation_results.json`：结构化评分结果。
- `evaluation_scores.csv`：各角度评分表。

跟踪视频默认保存到：

```text
~/Pictures/T/T1.avi
~/Pictures/T/T2.avi
...
```

如果 MediaMTX 正在运行，程序会推送两个 RTSP 流：

```text
rtsp://localhost:8554/live_video
rtsp://localhost:8554/best_frame
```

## 图像质量评价

程序使用五项指标计算综合评分：

```text
综合得分 = 0.40 * 亮度
        + 0.25 * 对比度
        + 0.15 * 色彩
        + 0.12 * 细节
        + 0.08 * 眩光抑制
```

这种评分方式避免只根据亮度选角，而是同时考虑曝光、纹理、颜色和高光抑制效果，更适合强反光环境下的偏振角寻优。

## 已记录结果

参赛文档中记录了多轮自动采集和跟踪对比结果，例如：

- `Capture_001`：最佳角度 40°，综合得分 91.23。
- `Capture_002`：最佳角度 10°，综合得分 82.54。
- `Capture_003`：最佳角度 150°，综合得分 83.27。

文档还记录了强反光场景下的跟踪效果对比：普通 RGB 输入的跟踪 IoU 为 48.5%，经过自适应偏振增强后提升到 67.5%。

## 注意事项

- 本项目依赖 RDK X5 的 BPU 推理环境和实际硬件，不适合直接在普通 Windows 或桌面 Python 环境中运行。
- 程序使用 OpenCV GUI 窗口，需要本地显示环境或可用的图形转发环境。
- 如果没有检测到 MediaMTX，RTSP 推流会自动禁用，不影响本地扫描、检测、跟踪和录制。
- 模型路径、串口列表、相机分辨率、录制格式和评分权重都集中定义在 Python 文件顶部，可按实际硬件修改。
