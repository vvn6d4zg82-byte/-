import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import serial
import time
import math
from collections import deque
from collections import Counter

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
    print("未找到机械臂")

def move(s, a):
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if cap.isOpened():
    print(f"摄像头已打开")
    cap.set(3, 640)
    cap.set(4, 480)
else:
    print("错误: 无可用摄像头")
    exit(1)

CUBE_COLORS = {
    'red':    (0, 0, 255),
    'white':  (255, 255, 255),
    'orange': (0, 140, 255),
    'green':  (0, 200, 0),
    'blue':   (255, 0, 0),
    'yellow': (0, 255, 255)
}

COLOR_NAMES = {
    'red': '红', 'white': '白', 'orange': '橙',
    'green': '绿', 'blue': '蓝', 'yellow': '黄'
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
        CUBE_COLORS['white'], CUBE_COLORS['yellow'],
        CUBE_COLORS['red'],   CUBE_COLORS['orange'],
        CUBE_COLORS['green'], CUBE_COLORS['blue']
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

def detect_cube_face(frame, cx, cy, size=60):
    roi = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    if roi.size == 0:
        return None
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_h = np.mean(hsv[:,:,0])
    avg_s = np.mean(hsv[:,:,1])
    avg_v = np.mean(hsv[:,:,2])
    
    if avg_s < 30:
        if avg_v > 180:
            return 'white'
        return 'white'
    
    if 0 <= avg_h <= 15 or avg_h > 170:
        if avg_s > 100:
            return 'red'
        return 'orange'
    elif 15 < avg_h <= 35:
        if avg_v > 180 and avg_s < 100:
            return 'yellow'
        return 'orange'
    elif 35 < avg_h <= 80:
        return 'green'
    elif 80 < avg_h <= 130:
        return 'blue'
    
    return None

def detect_cube_faces(frame, cube_cx, cube_cy, radius=80):
    faces = {}
    positions = [
        ('front', cube_cx, cube_cy),
        ('back', cube_cx, cube_cy - radius),
        ('left', cube_cx - radius, cube_cy),
        ('right', cube_cx + radius, cube_cy),
        ('top', cube_cx, cube_cy - radius),
        ('bottom', cube_cx, cube_cy + radius),
    ]
    
    for name, x, y in positions:
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            color = detect_cube_face(frame, x, y, 50)
            if color:
                faces[name] = color
    
    return faces

def apply_ml_stabilizer(detected_faces, history_buffer, stability_count=5):
    if len(history_buffer) < stability_count:
        history_buffer.append(detected_faces.copy())
        return detected_faces
    
    history_buffer.append(detected_faces.copy())
    if len(history_buffer) > stability_count:
        history_buffer.pop(0)
    
    stabilized = {}
    for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        values = [h.get(face_name) for h in history_buffer if face_name in h]
        if values:
            counter = Counter(values)
            stabilized[face_name] = counter.most_common(1)[0][0]
    
    return stabilized

def infer_rotation(prev_faces, curr_faces):
    if not prev_faces or not curr_faces:
        return 0, 0
    
    dx, dy = 0, 0
    
    front_moved_right = curr_faces.get('left') == prev_faces.get('front')
    front_moved_left = curr_faces.get('right') == prev_faces.get('front')
    
    if front_moved_right:
        dx = -0.08
    elif front_moved_left:
        dx = 0.08
    
    front_moved_up = curr_faces.get('bottom') == prev_faces.get('front')
    front_moved_down = curr_faces.get('top') == prev_faces.get('front')
    
    if front_moved_up:
        dy = -0.08
    elif front_moved_down:
        dy = 0.08
    
    return dx, dy

angles = [0.0, 0.0, 0.0]
face_history = deque(maxlen=8)
smooth_dx, smooth_dy = 0.0, 0.0
prev_faces = None
last_time = time.time()
frame_count = 0

print("=" * 50)
print("  魔方颜色同步控制器 v3 (ML稳定)")
print("=" * 50)
print("  旋转实体魔方 → 虚拟魔方同步跟随")
print("  空格键       → 归零")
print("  Q键          → 退出")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1
    
    cube_cx, cube_cy = w - 130, h // 2
    
    detected_faces = detect_cube_faces(frame, cube_cx, cube_cy, radius=70)
    
    stabilized_faces = apply_ml_stabilizer(detected_faces, face_history, stability_count=5)
    
    if prev_faces is not None:
        dx, dy = infer_rotation(prev_faces, stabilized_faces)
        
        alpha = 0.85
        smooth_dx = smooth_dx * alpha + dx * (1 - alpha)
        smooth_dy = smooth_dy * alpha + dy * (1 - alpha)
        
        if abs(smooth_dx) > 0.001:
            angles[1] += smooth_dx * 1.2
        if abs(smooth_dy) > 0.001:
            angles[0] += smooth_dy * 1.2
    
    prev_faces = stabilized_faces.copy() if stabilized_faces else None
    
    angles[0] = max(-math.pi/2, min(math.pi/2, angles[0]))
    angles[1] = max(-math.pi, min(math.pi, angles[1]))
    
    draw_cube(frame, angles, cube_cx, cube_cy)
    
    face_colors = {
        'front': CUBE_COLORS['blue'],
        'back': CUBE_COLORS['green'],
        'left': CUBE_COLORS['orange'],
        'right': CUBE_COLORS['red'],
        'top': CUBE_COLORS['white'],
        'bottom': CUBE_COLORS['yellow']
    }
    
    for i, (name, color) in enumerate(face_colors.items()):
        row = i // 2
        col = i % 2
        x = cube_cx - 50 + col * 80
        y = cube_cy + 110 + row * 35
        cv2.circle(frame, (x, y), 12, color, -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)
        name_cn = COLOR_NAMES.get(name, name) if name in CUBE_COLORS.values() else name
        color_name = stabilized_faces.get(name, "-")
        cv2.putText(frame, f"{name}:{color_name[:1] if color_name else '-'}",
                    (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    servo1 = int(90 + math.degrees(angles[1]) * 0.5)
    servo2 = int(90 + math.degrees(angles[0]) * 0.5)
    servo1 = max(0, min(180, servo1))
    servo2 = max(0, min(180, servo2))
    
    if time.time() - last_time > 0.016:
        if ser:
            move(1, servo1)
            move(2, servo2)
        last_time = time.time()
    
    cv2.rectangle(frame, (5, 5), (280, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (280, 120), (80, 80, 80), 1)
    
    cv2.putText(frame, f"检测面:{len(stabilized_faces)}/6", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"X:{math.degrees(angles[0]):.0f} Y:{math.degrees(angles[1]):.0f}", 
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"舵机1:{servo1} 舵机2:{servo2}", 
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
    cv2.putText(frame, f"FPS:{frame_count//max(int(time.time()-last_time+1),1)}", 
                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord(' '):
        angles = [0.0, 0.0, 0.0]
        face_history.clear()
        smooth_dx, smooth_dy = 0.0, 0.0
        prev_faces = None
        print("[重置]")
    
    cv2.imshow("魔方颜色同步", frame)

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("程序结束")
