#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
齿轮角度捕获与视频录制程序 - RDK X5 BPU加速版
集成 YOLOv8n(BPU) + DaSiamRPN 单目标跟踪（持续检测模式，可点击选择）
新增：跟踪模式下按 C 键保存跟踪视频（含边界框）
"""

import sys
sys.path.append('/home/sunrise/rdk_model_zoo')
sys.path.append('/home/sunrise/rdk_model_zoo/samples/vision/ultralytics_yolo/runtime/python')

import os
import time
import json
import csv
import threading
import queue
import signal
import atexit
import subprocess
import cv2
import numpy as np
import serial
from types import SimpleNamespace

from hobot_dnn import pyeasy_dnn as dnn
from ultralytics_yolo_det import UltralyticsYOLODetect

# ====================== 参数设置 ======================
BASE_DIR = os.path.expanduser('~/Pictures/gear_capture')
TRACK_VIDEO_DIR = os.path.expanduser('~/Pictures/T')

VIDEO_CODEC = 'XVID'
VIDEO_EXT = '.avi'
RECORD_VIDEO = True
RECORD_FPS = 30

DISPLAY_OSD = True
SAVE_EACH_STREAM_FRAME = False
SAVE_KEYFRAME_PER_ANGLE = True

LOCAL_WIN_W, LOCAL_WIN_H = 960, 540
STREAM_W, STREAM_H = 640, 480
LIVE_POS = (50, 50)
BEST_POS = (50 + LOCAL_WIN_W + 40, 50)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 1024
CAMERA_FPS = 30

RTSP_PORT = 8554
RTSP_PATHS = ['live_video', 'best_frame']

video_queues = [queue.Queue(maxsize=30), queue.Queue(maxsize=30)]
stream_active = True

best_frame_data = {'frame': None, 'angle': 0, 'score': 0.0, 'timestamp': time.time()}
rtsp_enabled = False

# ====================== BPU 模型路径 ======================
BPU_MODEL_PATH = os.path.expanduser(
    '~/rdk_model_zoo/samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin'
)
if not os.path.exists(BPU_MODEL_PATH):
    print(f"错误：BPU模型文件不存在: {BPU_MODEL_PATH}")
    print("请先下载模型到该路径")
    sys.exit(1)

cfg = SimpleNamespace(
    model_path=BPU_MODEL_PATH,
    classes_num=80,
    score_thres=0.3,
    nms_thres=0.45,
    resize_type=1,
    reg=16,
    strides=[8, 16, 32],
    topk=5,
    mc=32,
    nkpts=17,
    kpt_conf_thres=0.5,
)

try:
    detector = UltralyticsYOLODetect(cfg)
    print(f"BPU检测器初始化成功：{BPU_MODEL_PATH}")
    print("cfg 配置:", cfg)
except Exception as e:
    print(f"BPU检测器初始化失败: {e}")
    sys.exit(1)

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ====================== 图像质量评估（完全保留） ======================
POLARIZER_WEIGHTS = {
    'brightness': 0.40,
    'contrast': 0.25,
    'color': 0.15,
    'detail': 0.12,
    'glare_control': 0.08
}

def evaluate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    target_min, target_max = 50, 100
    if mean_brightness < target_min:
        brightness_score = max(0, 100 - (target_min - mean_brightness) * 2)
    elif mean_brightness > target_max:
        brightness_score = max(0, 100 - (mean_brightness - target_max) * 0.5)
    else:
        brightness_score = 100
    uniformity_score = max(0, 100 - brightness_std * 0.5)
    final_score = brightness_score * 0.7 + uniformity_score * 0.3
    return {'final_score': final_score}

def evaluate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    if laplacian_var > 1000:
        contrast_score = 100
    elif laplacian_var < 100:
        contrast_score = laplacian_var
    else:
        contrast_score = min(100, laplacian_var * 0.1)
    return {'final_score': contrast_score}

def evaluate_color_saturation(frame):
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    mean_sat = np.mean(saturation)
    if mean_sat < 100:
        sat_score = mean_sat
    elif mean_sat > 180:
        sat_score = max(0, 100 - (mean_sat - 180) * 0.5)
    else:
        sat_score = 100
    return {'final_score': sat_score}

def evaluate_detail(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block_size = 32
    energies = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            if block.size:
                energies.append(np.var(block))
    mean_energy = np.mean(energies) if energies else 0
    if mean_energy > 50:
        detail_score = 100
    elif mean_energy < 10:
        detail_score = mean_energy * 10
    else:
        detail_score = min(100, mean_energy * 2)
    return {'final_score': detail_score}

def evaluate_glare_control(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    highlight_ratio = np.sum(gray > 220) / gray.size * 100
    overexposed_ratio = np.sum(gray == 255) / gray.size * 100
    highlight_penalty = max(0, 100 - highlight_ratio * 2)
    overexposure_penalty = max(0, 100 - overexposed_ratio * 5)
    final_score = highlight_penalty * 0.6 + overexposure_penalty * 0.4
    return {'final_score': final_score}

def comprehensive_image_evaluation(frame, angle):
    results = {'angle': angle}
    for cat, func in [
        ('brightness', evaluate_brightness),
        ('contrast', evaluate_contrast),
        ('color', evaluate_color_saturation),
        ('detail', evaluate_detail),
        ('glare_control', evaluate_glare_control)
    ]:
        results[cat] = func(frame)
    overall = sum(results[cat]['final_score'] * POLARIZER_WEIGHTS[cat] for cat in POLARIZER_WEIGHTS)
    results['overall_score'] = overall
    return results

def update_best_frame(frame, angle, score):
    global best_frame_data
    if best_frame_data['frame'] is None or score > best_frame_data['score']:
        best_frame_data['frame'] = frame.copy()
        best_frame_data['angle'] = angle
        best_frame_data['score'] = score
        best_frame_data['timestamp'] = time.time()
        print(f"最佳帧已更新: 角度 {angle}°, 得分 {score:.1f}")

def get_best_frame_for_display():
    if best_frame_data['frame'] is None:
        frame = np.zeros((LOCAL_WIN_H, LOCAL_WIN_W, 3), dtype=np.uint8)
        cv2.putText(frame, "Waiting for best frame...", (50, LOCAL_WIN_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return frame
    frame = cv2.resize(best_frame_data['frame'], (LOCAL_WIN_W, LOCAL_WIN_H))
    lines = [
        f"Best Angle: {best_frame_data['angle']}°",
        f"Overall Score: {best_frame_data['score']:.1f}/100",
        f"Updated: {time.strftime('%H:%M:%S', time.localtime(best_frame_data['timestamp']))}",
        "Press 'A' next round | ESC exit"
    ]
    y = 40
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 35
    cv2.putText(frame, f"Score: {best_frame_data['score']:.1f}", (LOCAL_WIN_W-200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    return frame

def print_evaluation_summary(eval_dict):
    print("\n" + "="*60)
    print(f"角度: {eval_dict['angle']}°  综合评分: {eval_dict['overall_score']:.1f}/100")
    for cat in POLARIZER_WEIGHTS:
        print(f"{cat}: {eval_dict[cat]['final_score']:.1f}")
    print("="*60)

# ====================== 视频流处理器 ======================
class VideoStreamProcessor:
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
        self.process = None
        self.running = False
        self.thread = None

    def start(self):
        if not rtsp_enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self._cleanup()

    def _setup_ffmpeg(self):
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{STREAM_W}x{STREAM_H}', '-r', '15', '-i', '-',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                '-pix_fmt', 'yuv420p', '-b:v', '800k', '-maxrate', '1200k',
                '-bufsize', '2000k', '-g', '30', '-keyint_min', '30',
                '-f', 'rtsp', '-rtsp_transport', 'tcp', '-muxdelay', '0.1',
                '-loglevel', 'error', f'rtsp://localhost:8554/{RTSP_PATHS[self.idx]}'
            ]
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, bufsize=10**7)
            time.sleep(1)
            return self.process.poll() is None
        except:
            return False

    def _cleanup(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                pass
            self.process = None

    def _worker(self):
        while self.running:
            if self.process is None or self.process.poll() is not None:
                self._cleanup()
                time.sleep(1)
                self._setup_ffmpeg()
                continue
            try:
                frame = video_queues[self.idx].get(timeout=0.5)
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
            except queue.Empty:
                pass
            except:
                time.sleep(0.5)
        self._cleanup()

stream_processors = []

def check_mediamtx():
    try:
        import socket
        s = socket.socket()
        s.settimeout(2)
        s.connect(('localhost', 8554))
        s.close()
        return True
    except:
        return False

def setup_video_streams():
    global rtsp_enabled, stream_processors
    if not check_mediamtx():
        rtsp_enabled = False
        print("Mediamtx 未运行，RTSP 流已禁用")
        return True
    rtsp_enabled = True
    for i in range(2):
        name = "实时视频" if i == 0 else "最佳帧"
        proc = VideoStreamProcessor(i, name)
        proc.start()
        stream_processors.append(proc)
    return True

def put_frame_to_stream(frame, idx):
    if not stream_active or not rtsp_enabled:
        return
    try:
        if video_queues[idx].full():
            video_queues[idx].get_nowait()
        video_queues[idx].put_nowait(cv2.resize(frame, (STREAM_W, STREAM_H)))
    except:
        pass

def cleanup_video_streams():
    global stream_active
    stream_active = False
    for p in stream_processors:
        p.stop()
    stream_processors.clear()

# ====================== 串口控制 ======================
class GearController:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False

    def connect(self):
        ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyAMA0', '/dev/ttyS0', '/dev/ttyTHS1']
        for p in ports:
            try:
                self.ser = serial.Serial(p, self.baud, timeout=1)
                print(f"串口已连接: {p}")
                self.connected = True
                time.sleep(2)
                return True
            except:
                continue
        return False

    def send_command(self, cmd):
        if not self.connected or self.ser is None:
            return False
        try:
            self.ser.write(f"{cmd}\n".encode())
            self.ser.flush()
            print(f"命令发送: {cmd}")
            return True
        except:
            self.connected = False
            return False

    def disconnect(self):
        if self.ser:
            self.ser.close()
            self.ser = None
        self.connected = False

# ====================== 辅助函数 ======================
def put_osd_for_display(img, lines, org=(10,28), scale=0.7):
    if not DISPLAY_OSD or img is None:
        return img
    out = img.copy()
    y = org[1]
    for line in lines:
        cv2.putText(out, line, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), 2)
        y += int(30 * scale)
    return out

def create_round_directory():
    existing = []
    for d in os.listdir(BASE_DIR):
        if d.startswith("Capture_") and os.path.isdir(os.path.join(BASE_DIR, d)):
            try:
                existing.append(int(d.split('_')[1]))
            except:
                pass
    next_num = max(existing) + 1 if existing else 1
    round_dir = os.path.join(BASE_DIR, f"Capture_{next_num:03d}")
    os.makedirs(round_dir, exist_ok=True)
    frames_dir = os.path.join(round_dir, "stream_frames") if SAVE_EACH_STREAM_FRAME else None
    keyframes_dir = os.path.join(round_dir, "angle_keyframes") if SAVE_KEYFRAME_PER_ANGLE else None
    for d in [frames_dir, keyframes_dir]:
        if d:
            os.makedirs(d, exist_ok=True)
    return next_num, round_dir, frames_dir, keyframes_dir

class CameraManager:
    def __init__(self):
        self.cap = None

    def initialize(self):
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            fourcc_mjpg = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            time.sleep(0.2)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = chr(actual_fourcc & 0xFF) + chr((actual_fourcc >> 8) & 0xFF) + \
                         chr((actual_fourcc >> 16) & 0xFF) + chr((actual_fourcc >> 24) & 0xFF)
            print(f"相机 {idx}: {actual_width}x{actual_height} @ {actual_fps:.1f}fps, 编码: {fourcc_str}")
            self.cap = cap
            return True
        return False

    def read_frame(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def resize_for_local_display(self, frame):
        return cv2.resize(frame, (LOCAL_WIN_W, LOCAL_WIN_H))

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# ====================== 全局线程 ======================
latest_angle = 0.0
latest_frame = None
exit_flag = False

def serial_thread_func(ser):
    global latest_angle, exit_flag
    while not exit_flag:
        try:
            if ser and ser.is_open:
                line = ser.readline().decode(errors='ignore').strip()
                if line.startswith("ANG:"):
                    try:
                        latest_angle = float(line.split(':')[1])
                    except:
                        pass
        except:
            pass
        time.sleep(0.001)

def camera_thread_func(cap):
    global latest_frame, exit_flag
    while not exit_flag:
        try:
            if cap:
                ret, frame = cap.read()
                if ret:
                    latest_frame = frame.copy()
        except:
            pass
        time.sleep(0.005)

# ====================== 录制管理器 ======================
class RecordingManager:
    def __init__(self, camera, gear):
        self.camera = camera
        self.gear = gear

    def record_one_round(self):
        # 此函数与ByteTrack版本完全相同，保留
        round_start_time = time.time()
        print("\n========== 开始新的一轮 ==========")
        if not self.gear.connected:
            print("串口未连接")
            return False

        print("归零中...")
        self.gear.send_command("T0")
        zero_timeout = time.time() + 8.0
        while time.time() < zero_timeout:
            cur_angle = latest_angle
            if abs(cur_angle) < 1.5:
                print(f"归零完成，当前角度: {cur_angle:.1f}°")
                break
            time.sleep(0.01)
        else:
            print(f"警告：归零超时，当前角度 {latest_angle:.1f}°，继续执行")

        round_num, round_dir, _, _ = create_round_directory()
        print(f"\n第 {round_num} 轮开始")

        video_writer = None
        video_path = None
        if RECORD_VIDEO:
            ret, test = self.camera.read_frame()
            h, w = test.shape[:2] if ret else (CAMERA_HEIGHT, CAMERA_WIDTH)
            video_path = os.path.join(round_dir, f"capture_{round_num:03d}{VIDEO_EXT}")
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            video_writer = cv2.VideoWriter(video_path, fourcc, RECORD_FPS, (w, h))
            if video_writer.isOpened():
                print(f"视频录制: {video_path} (帧率={RECORD_FPS}fps)")
            else:
                video_writer = None

        frame_cache = {}
        captured = set()
        targets = list(range(0, 181, 10))
        tolerance = 3.5
        crossed = {t: False for t in targets}
        prev_angle = latest_angle

        frm0 = latest_frame
        if frm0 is not None:
            cur_angle = latest_angle
            print(f"抓拍 目标=0° 当前={cur_angle:.1f}°")
            cv2.imwrite(os.path.join(round_dir, "angle_000.jpg"), frm0)
            frame_cache[0] = frm0.copy()
            captured.add(0)
            crossed[0] = True

        print("开始旋转...")
        self.gear.send_command("T180")

        start_time = time.time()
        timeout = 60

        while len(captured) < len(targets) and (time.time() - start_time) < timeout:
            cur_angle = latest_angle
            frm = latest_frame
            if frm is None:
                continue
            if video_writer:
                video_writer.write(frm)

            for t in targets:
                if t in captured:
                    continue
                if abs(cur_angle - t) <= tolerance:
                    print(f"抓拍 目标={t}° 当前={cur_angle:.1f}°")
                    cv2.imwrite(os.path.join(round_dir, f"angle_{t:03d}.jpg"), frm)
                    frame_cache[t] = frm.copy()
                    captured.add(t)
                    crossed[t] = True
                    continue
                if not crossed[t]:
                    if (prev_angle < t and cur_angle >= t) or (prev_angle > t and cur_angle <= t):
                        if abs(cur_angle - t) <= 5.0:
                            print(f"跨角度抓拍 目标={t}° 当前={cur_angle:.1f}°")
                            cv2.imwrite(os.path.join(round_dir, f"angle_{t:03d}.jpg"), frm)
                            frame_cache[t] = frm.copy()
                            captured.add(t)
                            crossed[t] = True
            prev_angle = cur_angle

            disp = self.camera.resize_for_local_display(frm)
            osd_lines = [f"Captured: {len(captured)}/{len(targets)}", f"Current angle: {cur_angle:.1f}°"]
            cv2.imshow('Live Video', put_osd_for_display(disp, osd_lines, scale=0.6))
            cv2.imshow('Best Frame', get_best_frame_for_display())
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("用户中断")
                if video_writer:
                    video_writer.release()
                return False

        if video_writer:
            video_writer.release()
            print(f"视频已保存: {video_path}")
        print(f"抓拍耗时: {time.time()-start_time:.2f}s")

        missing = [t for t in targets if t not in captured]
        if missing:
            print(f"警告：未抓拍角度 {missing}")

        print("\n开始图像质量评估...")
        eval_results = []
        for ang, frm in frame_cache.items():
            ev = comprehensive_image_evaluation(frm, ang)
            eval_results.append(ev)
            print(f"角度 {ang}° 得分: {ev['overall_score']:.1f}")

        if eval_results:
            best = max(eval_results, key=lambda x: x['overall_score'])
            best_angle = best['angle']
            best_score = best['overall_score']
            best_frame = frame_cache[best_angle]
            best_path = os.path.join(round_dir, f"best_{best_angle:03d}deg_{best_score:.1f}.jpg")
            cv2.imwrite(best_path, best_frame)
            print(f"最佳帧: {best_path}")
            update_best_frame(best_frame, best_angle, best_score)
            print_evaluation_summary(best)

            try:
                with open(os.path.join(round_dir, "evaluation_results.json"), 'w') as f:
                    json.dump([{
                        'angle': e['angle'],
                        'overall_score': e['overall_score'],
                        **{cat: e[cat]['final_score'] for cat in POLARIZER_WEIGHTS}
                    } for e in eval_results], f, indent=2)
                with open(os.path.join(round_dir, "evaluation_scores.csv"), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['angle','overall','brightness','contrast','color','detail','glare'])
                    for e in eval_results:
                        writer.writerow([
                            e['angle'], f"{e['overall_score']:.1f}",
                            f"{e['brightness']['final_score']:.1f}",
                            f"{e['contrast']['final_score']:.1f}",
                            f"{e['color']['final_score']:.1f}",
                            f"{e['detail']['final_score']:.1f}",
                            f"{e['glare_control']['final_score']:.1f}"
                        ])
            except Exception as e:
                print(f"保存评估数据失败: {e}")
        else:
            print("未收集到评估数据")

        if 'best_angle' in locals() and best_angle is not None:
            print(f"返回最佳角度: {best_angle}°")
            self.gear.send_command(f"T{best_angle}")
        total_time = time.time() - round_start_time
        print(f"【本轮总用时】: {total_time:.2f} 秒 (从归零开始到发送返回命令)")
        return True

# ====================== 主函数 ======================
def main():
    global exit_flag, latest_frame, latest_angle

    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_resources)
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(TRACK_VIDEO_DIR, exist_ok=True)

    print("="*70)
    print("偏振镜控制系统 + YOLOv8n(BPU) + DaSiamRPN 单目标跟踪")
    print("RDK X5 移植版 - 使用BPU加速 + DaSiamRPN跟踪")
    print("新增功能：跟踪模式下按 C 键开始/停止保存跟踪视频（含边界框）")
    print("模式说明：")
    print("  - SCAN: 按 A 启动偏振扫描，扫描后自动回到 SCAN")
    print("  - SCAN 状态下按 T：进入目标检测模式（持续检测，每帧检测）")
    print("  - 检测模式下点击任意框选择目标，切换到跟踪模式（高亮该目标）")
    print("  - 跟踪时按 A 或 X 停止跟踪返回 SCAN，按 C 开始/停止录制跟踪视频")
    print("  - ESC 退出程序")
    print("="*70)

    cam = CameraManager()
    if not cam.initialize():
        print("相机初始化失败")
        return

    gear = GearController()
    if not gear.connect():
        print("串口连接失败")
        cam.release()
        return

    exit_flag = False
    threading.Thread(target=serial_thread_func, args=(gear.ser,), daemon=True).start()
    threading.Thread(target=camera_thread_func, args=(cam.cap,), daemon=True).start()
    time.sleep(0.5)

    setup_video_streams()

    cv2.namedWindow('Live Video')
    cv2.namedWindow('Best Frame')
    cv2.moveWindow('Live Video', *LIVE_POS)
    cv2.moveWindow('Best Frame', *BEST_POS)

    recorder = RecordingManager(cam, gear)

    MODE_SCAN = 0
    MODE_TRACK_SELECT = 1
    MODE_TRACKING = 2
    current_mode = MODE_SCAN

    # ==================== DaSiamRPN 跟踪器 ====================
    tracker = None          # 用于存储 cv2.TrackerDaSiamRPN 实例
    tracking_bbox = None    # 当前跟踪框 (x, y, w, h)

    # 持续检测相关变量（保留YOLO检测，用于目标选择）
    detection_running = False
    detection_thread = None
    detection_lock = threading.Lock()
    latest_detection = {
        'boxes': ([], [], []),   # (boxes_scaled, orig_boxes, labels)
        'track_ids': []          # 不再使用，但保留结构
    }
    frame_counter = 0
    DETECT_EVERY_N_FRAMES = 1   # 每帧检测

    # 跟踪视频录制相关变量
    track_video_writer = None
    track_video_recording = False
    track_video_counter = 1

    def get_next_track_video_number():
        if not os.path.exists(TRACK_VIDEO_DIR):
            return 1
        existing = []
        for f in os.listdir(TRACK_VIDEO_DIR):
            if f.startswith("T") and f.endswith(VIDEO_EXT):
                try:
                    num = int(f[1:-4])
                    existing.append(num)
                except:
                    pass
        return max(existing) + 1 if existing else 1

    def start_track_recording(frame):
        nonlocal track_video_writer, track_video_recording, track_video_counter
        if track_video_recording:
            print("已在录制中，请先停止当前录制")
            return
        num = get_next_track_video_number()
        filename = os.path.join(TRACK_VIDEO_DIR, f"T{num}{VIDEO_EXT}")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(filename, fourcc, RECORD_FPS, (w, h))
        if writer.isOpened():
            track_video_writer = writer
            track_video_recording = True
            track_video_counter = num + 1
            print(f"开始录制跟踪视频: {filename}")
        else:
            print("无法创建视频文件")

    def stop_track_recording():
        nonlocal track_video_writer, track_video_recording
        if track_video_writer is not None:
            track_video_writer.release()
            track_video_writer = None
        track_video_recording = False
        print("停止录制跟踪视频")

    select_data = {'boxes_orig': [], 'scale_x': 1.0, 'scale_y': 1.0}

    # ==================== 鼠标回调：点击选择目标 ====================
    def mouse_callback(event, x, y, flags, param):
        nonlocal tracker, tracking_bbox, current_mode
        if current_mode != MODE_TRACK_SELECT:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            boxes = param.get('boxes_orig', [])
            if not boxes:
                return
            scale_x = param.get('scale_x', 1.0)
            scale_y = param.get('scale_y', 1.0)
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            for box in boxes:
                x1, y1, x2, y2 = box
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    # 初始化DaSiamRPN跟踪器
                    try:
                        # 尝试创建DaSiamRPN
                        tracker = cv2.TrackerDaSiamRPN_create()
                    except AttributeError:
                        # 备选：使用CSRT（相关滤波，鲁棒性较好）
                        print("DaSiamRPN不可用，使用CSRT跟踪器")
                        tracker = cv2.TrackerCSRT_create()
                    bbox = (x1, y1, x2-x1, y2-y1)
                    tracker.init(latest_frame, bbox)
                    tracking_bbox = bbox
                    print(f"选中目标框: {bbox}")
                    current_mode = MODE_TRACKING
                    break

    cv2.setMouseCallback('Live Video', mouse_callback, select_data)

    # ==================== 检测线程（仅用于目标选择，仍保留YOLO检测） ====================
    def detection_worker():
        nonlocal latest_detection, detection_running
        while detection_running:
            time.sleep(0.01)
            with detection_lock:
                frame_to_detect = latest_frame
                if frame_to_detect is None:
                    continue
                frm = frame_to_detect.copy()
            try:
                boxes, scores, class_ids = detector.predict(frm)

                orig_boxes = []
                labels = []
                if len(boxes) > 0:
                    for box, score, cls_id in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1 = max(0, min(frm.shape[1], x1))
                        y1 = max(0, min(frm.shape[0], y1))
                        x2 = max(0, min(frm.shape[1], x2))
                        y2 = max(0, min(frm.shape[0], y2))
                        orig_boxes.append((x1, y1, x2, y2))
                        labels.append(f"{COCO_NAMES[cls_id]} {score:.2f}")

                scale_x_display = LOCAL_WIN_W / frm.shape[1]
                scale_y_display = LOCAL_WIN_H / frm.shape[0]
                boxes_scaled = []
                for box in orig_boxes:
                    x1, y1, x2, y2 = box
                    boxes_scaled.append((
                        int(x1 * scale_x_display),
                        int(y1 * scale_y_display),
                        int(x2 * scale_x_display),
                        int(y2 * scale_y_display)
                    ))

                with detection_lock:
                    latest_detection['boxes'] = (boxes_scaled, orig_boxes, labels)
                    latest_detection['track_ids'] = []  # 不再使用

            except Exception as e:
                print(f"检测线程错误: {e}")

    # ==================== 主循环 ====================
    while not exit_flag:
        frame = latest_frame
        if frame is None:
            time.sleep(0.005)
            continue

        if current_mode == MODE_SCAN:
            disp = cam.resize_for_local_display(frame)
            osd_lines = ["SCAN MODE - Press 'A' to start polarization scan",
                         "Press 'T' to enter target detection mode"]
            cv2.imshow('Live Video', put_osd_for_display(disp, osd_lines, scale=0.6))
            cv2.imshow('Best Frame', get_best_frame_for_display())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            elif key == ord('a') or key == ord('A'):
                print("启动偏振扫描...")
                try:
                    recorder.record_one_round()
                except Exception as e:
                    print(f"扫描异常: {e}")
                current_mode = MODE_SCAN
            elif key == ord('t') or key == ord('T'):
                print("进入持续目标检测模式...")
                detection_running = True
                frame_counter = 0
                latest_detection = {'boxes': ([], [], []), 'track_ids': []}
                detection_thread = threading.Thread(target=detection_worker, daemon=True)
                detection_thread.start()
                current_mode = MODE_TRACK_SELECT

        elif current_mode == MODE_TRACK_SELECT:
            with detection_lock:
                boxes_scaled, orig_boxes, labels = latest_detection['boxes']
            select_data['boxes_orig'] = orig_boxes
            select_data['scale_x'] = frame.shape[1] / LOCAL_WIN_W
            select_data['scale_y'] = frame.shape[0] / LOCAL_WIN_H

            display_img = cv2.resize(frame, (LOCAL_WIN_W, LOCAL_WIN_H))
            for i, box in enumerate(boxes_scaled):
                x1, y1, x2, y2 = box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0,255,0), 2)
                if i < len(labels):
                    label = labels[i]
                    cv2.putText(display_img, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(display_img, "DETECT MODE - Click on a box to track, press X to cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.imshow('Live Video', display_img)
            cv2.imshow('Best Frame', get_best_frame_for_display())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            elif key == ord('x') or key == ord('X'):
                print("退出检测模式")
                detection_running = False
                if detection_thread:
                    detection_thread.join(timeout=1)
                current_mode = MODE_SCAN
                tracker = None
                tracking_bbox = None
                continue

        elif current_mode == MODE_TRACKING:
            # 执行跟踪
            if tracker is None:
                print("跟踪器未初始化，返回扫描模式")
                current_mode = MODE_SCAN
                continue

            success, bbox = tracker.update(frame)
            disp = cam.resize_for_local_display(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                tracking_bbox = (x, y, w, h)
                # 绘制跟踪框
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.putText(disp, "Target", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # 录制视频
                if track_video_recording and track_video_writer is not None:
                    orig_frame_with_track = frame.copy()
                    cv2.rectangle(orig_frame_with_track, (x, y), (x+w, y+h), (0, 255, 255), 3)
                    cv2.putText(orig_frame_with_track, "Target", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    track_video_writer.write(orig_frame_with_track)
            else:
                # 跟踪丢失
                print("目标丢失，退出跟踪模式")
                tracker = None
                tracking_bbox = None
                current_mode = MODE_TRACK_SELECT
                continue

            if track_video_recording:
                cv2.putText(disp, "REC", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            osd_lines = ["TRACKING MODE - Press 'A' or 'X' to stop tracking",
                         "Press 'C' to start/stop recording video"]
            cv2.imshow('Live Video', put_osd_for_display(disp, osd_lines, scale=0.6))
            cv2.imshow('Best Frame', get_best_frame_for_display())

            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            elif key == ord('a') or key == ord('A') or key == ord('x') or key == ord('X'):
                print("停止跟踪，返回扫描模式")
                if track_video_recording:
                    stop_track_recording()
                current_mode = MODE_SCAN
                tracker = None
                tracking_bbox = None
                detection_running = False
                if detection_thread:
                    detection_thread.join(timeout=1)
            elif key == ord('c') or key == ord('C'):
                if not track_video_recording:
                    start_track_recording(frame)
                else:
                    stop_track_recording()

        else:
            current_mode = MODE_SCAN

        if current_mode == MODE_TRACK_SELECT:
            frame_counter += 1

    cleanup_resources()

def cleanup_resources():
    global exit_flag
    exit_flag = True
    time.sleep(0.2)
    cleanup_video_streams()
    cv2.destroyAllWindows()
    print("程序已退出")

def signal_handler(sig, frame):
    print("\n收到中断信号，正在退出...")
    cleanup_resources()
    sys.exit(0)

if __name__ == "__main__":
    main()