import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import serial
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as image_module
import math

try:
    for port in ['COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM3', 'COM4']:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"串口连接: {port}")
            time.sleep(2)
            break
        except:
            continue
    else:
        ser = None
except:
    ser = None

if ser:
    print("机械臂已连接")
else:
    print("未找到机械臂(串口)，仅运行虚拟模式")

def move(s, a):
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"错误: 找不到模型 {model_path}")
    exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
for device_id in [0, 1]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None:
            print(f"摄像头: {device_id}")
            break
else:
    print("错误: 无可用摄像头")
    exit(1)

cap.set(3, 640)
cap.set(4, 480)

CUBE_COLORS = {
    'red':    (0, 0, 255),
    'white':  (255, 255, 255),
    'orange': (0, 140, 255),
    'green':  (0, 200, 0),
    'blue':   (255, 0, 0),
    'yellow': (0, 255, 255)
}

def rotate_point(x, y, z, ax, ay, az):
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    
    y1 = y * cx - z * sx
    z1 = y * sx + z * cx
    x2 = x * cy + z1 * sy
    z2 = -x * sy + z1 * cy
    x3 = x2 * cz - y1 * sz
    y3 = x2 * sz + y1 * cz
    return x3, y3, z2

def project(x, y, z, cx, cy, s, fov=500):
    z += 3
    f = fov / (fov + z * 80)
    return int(cx + x * s * f), int(cy + y * s * f)

def draw_cube(frame, angles, cx, cy, scale=75):
    colors = [
        CUBE_COLORS['white'],
        CUBE_COLORS['yellow'],
        CUBE_COLORS['red'],
        CUBE_COLORS['orange'],
        CUBE_COLORS['green'],
        CUBE_COLORS['blue']
    ]
    
    verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vx, vy, vz = rotate_point(x, y, z, angles[0], angles[1], angles[2])
                verts.append((vx, vy, vz))
    
    faces = [
        ([0,1,3,2], 0),
        ([4,5,7,6], 1),
        ([0,4,6,2], 2),
        ([1,5,7,3], 3),
        ([0,1,5,4], 4),
        ([2,3,7,6], 5)
    ]
    
    proj = [project(v[0], v[1], v[2], cx, cy, scale) for v in verts]
    
    z_list = [(sum(verts[j][2] for j in f) / 4, i) for f, i in faces]
    z_list.sort(key=lambda x: x[0])
    
    for _, i in z_list:
        idx, ci = faces[i]
        pts = np.array([proj[j] for j in idx], np.int32)
        cv2.fillPoly(frame, [pts], colors[ci])
        cv2.polylines(frame, [pts], True, (180, 180, 180), 1)
    
    return proj

angles = [0.0, 0.0, 0.0]
hand_pos = []
smooth_dx, smooth_dy = 0.0, 0.0
last_time = time.time()

print("=" * 50)
print("  魔方机械臂控制器 v2")
print("=" * 50)
print("  挥手左右 → Y轴旋转")
print("  挥手上下 → X轴旋转")
print("  空格键   → 归零")
print("  Q键      → 退出")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_module.Image(image_format=image_module.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)
    
    raw_dx, raw_dy = 0.0, 0.0
    
    if results and results.hand_landmarks:
        lm = results.hand_landmarks[0]
        wrist = lm[0]
        wx, wy = wrist.x, wrist.y
        hand_pos.append((wx, wy))
        if len(hand_pos) > 8:
            hand_pos.pop(0)
        
        if len(hand_pos) >= 3:
            delta = 3
            raw_dx = (hand_pos[-1][0] - hand_pos[-delta][0]) * 2.0
            raw_dy = (hand_pos[-1][1] - hand_pos[-delta][1]) * 2.0
        
        for pt in lm:
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    alpha = 0.7
    smooth_dx = smooth_dx * alpha + raw_dx * (1 - alpha)
    smooth_dy = smooth_dy * alpha + raw_dy * (1 - alpha)
    
    threshold = 0.01
    if abs(smooth_dx) > threshold:
        angles[1] += smooth_dx * 1.5
    if abs(smooth_dy) > threshold:
        angles[0] += smooth_dy * 1.5
    
    angles[0] = max(-math.pi/2, min(math.pi/2, angles[0]))
    angles[1] = max(-math.pi, min(math.pi, angles[1]))
    
    cube_cx, cube_cy = w - 130, h // 2
    draw_cube(frame, angles, cube_cx, cube_cy)
    
    for i, color_name in enumerate(['红', '白', '橙', '绿', '蓝', '明黄']):
        col = i % 3
        row = i // 3
        x = cube_cx - 60 + col * 50
        y = cube_cy + 100 + row * 40
        color_val = list(CUBE_COLORS.values())[i]
        cv2.circle(frame, (x, y), 12, color_val, -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)
        cv2.putText(frame, color_name, (x - 10, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    servo1 = int(90 + math.degrees(angles[1]) * 0.5)
    servo2 = int(90 + math.degrees(angles[0]) * 0.5)
    servo1 = max(0, min(180, servo1))
    servo2 = max(0, min(180, servo2))
    
    if time.time() - last_time > 0.05:
        if ser:
            move(1, servo1)
            move(2, servo2)
        last_time = time.time()
    
    cv2.rectangle(frame, (5, 5), (280, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (280, 100), (80, 80, 80), 1)
    
    status = "检测" if results and results.hand_landmarks else "未检测"
    color = (0, 255, 0) if status == "检测" else (0, 0, 255)
    cv2.putText(frame, f"手势: {status}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"角度 X:{math.degrees(angles[0]):.0f} Y:{math.degrees(angles[1]):.0f}", 
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"舵机1:{servo1} 舵机2:{servo2}", 
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
    
    key = cv2.waitKey(16) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord(' '):
        angles = [0.0, 0.0, 0.0]
        hand_pos = []
        smooth_dx, smooth_dy = 0.0, 0.0
        print("[重置]")
    
    cv2.imshow("魔方机械臂控制", frame)

cap.release()
cv2.destroyAllWindows()
detector.close()
if ser:
    ser.close()
print("程序结束")
