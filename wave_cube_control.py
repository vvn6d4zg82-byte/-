import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as image_module
import math
import time

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"错误: 找不到模型文件 {model_path}")
    exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
for device_id in [0, 1]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None:
            print(f"使用摄像头设备 {device_id}")
            break
else:
    print("错误: 没有可用的摄像头")
    exit(1)

cap.set(3, 640)
cap.set(4, 480)

CUBE_COLORS = {
    'white':  (255, 255, 255),
    'yellow': (0, 255, 255),
    'red':    (0, 0, 255),
    'orange': (0, 165, 255),
    'green':  (0, 255, 0),
    'blue':   (255, 0, 0)
}

def rotate_point(x, y, z, angle_x, angle_y, angle_z):
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)
    
    y1 = y * cx - z * sx
    z1 = y * sx + z * cx
    x2 = x * cy + z1 * sy
    z2 = -x * sy + z1 * cy
    x3 = x2 * cz - y1 * sz
    y3 = x2 * sz + y1 * cz
    
    return x3, y3, z2

def project_3d(x, y, z, cx, cy, scale, fov=500):
    z += 3
    factor = fov / (fov + z * 100)
    px = int(cx + x * scale * factor)
    py = int(cy + y * scale * factor)
    return px, py

def draw_3d_cube(frame, angles, cx, cy, scale=70):
    face_colors = [
        CUBE_COLORS['white'],
        CUBE_COLORS['yellow'],
        CUBE_COLORS['red'],
        CUBE_COLORS['orange'],
        CUBE_COLORS['green'],
        CUBE_COLORS['blue']
    ]
    
    vertices = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vx, vy, vz = rotate_point(x, y, z, angles[0], angles[1], angles[2])
                vertices.append((vx, vy, vz))
    
    faces = [
        ([0,1,3,2], 0),
        ([4,5,7,6], 1),
        ([0,4,6,2], 2),
        ([1,5,7,3], 3),
        ([0,1,5,4], 4),
        ([2,3,7,6], 5)
    ]
    
    projected = [project_3d(v[0], v[1], v[2], cx, cy, scale) for v in vertices]
    
    z_order = []
    for i, (face, _) in enumerate(faces):
        z_avg = sum(vertices[j][2] for j in face) / 4
        z_order.append((z_avg, i))
    z_order.sort()
    
    for z_avg, i in z_order:
        face_indices, color_idx = faces[i]
        pts = np.array([projected[j] for j in face_indices], np.int32)
        cv2.fillPoly(frame, [pts], face_colors[color_idx])
        cv2.polylines(frame, [pts], True, (200, 200, 200), 1)
    
    return projected

def detect_cube_color(frame, roi_x, roi_y, roi_w, roi_h):
    if roi_x < 0 or roi_y < 0 or roi_x + roi_w > frame.shape[1] or roi_y + roi_h > frame.shape[0]:
        return None
    
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    if roi.size == 0:
        return None
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'white':  ([0, 0, 200], [180, 30, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'red':    ([0, 100, 100], [10, 255, 255]),
        'orange': ([10, 100, 100], [20, 255, 255]),
        'green':  ([40, 50, 50], [80, 255, 255]),
        'blue':   ([100, 50, 50], [130, 255, 255])
    }
    
    max_pixels = 0
    detected = None
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = cv2.countNonZero(mask)
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected = color_name
    
    return detected if max_pixels > 100 else None

def detect_wave(positions, threshold=0.03):
    if len(positions) < 5:
        return 0, 0
    
    recent = positions[-5:]
    dx = sum(p[0] for p in recent) / len(recent)
    dy = sum(p[1] for p in recent) / len(recent)
    
    var_x = sum((p[0] - dx) ** 2 for p in recent)
    var_y = sum((p[1] - dy) ** 2 for p in recent)
    
    if var_x > threshold or var_y > threshold:
        return (dx - positions[0][0]) * 3, (dy - positions[0][1]) * 3
    
    return 0, 0

angles = [0.0, 0.0, 0.0]
hand_positions = []
last_wave_time = 0
last_reset_time = 0

print("=" * 50)
print("  挥手魔方控制器")
print("=" * 50)
print("控制说明:")
print("  挥手左右 → 魔方绕Y轴旋转")
print("  挥手上下 → 魔方绕X轴旋转")
print("  空格键  → 归零")
print("  Q键     → 退出")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    frame_show = frame.copy()
    h, w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_module.Image(image_format=image_module.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)
    
    hand_detected = False
    current_pos = None
    
    if results and results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        hand_detected = True
        
        wrist = landmarks[0]
        wrist_x = wrist.x
        wrist_y = wrist.y
        
        current_pos = (wrist_x, wrist_y)
        hand_positions.append(current_pos)
        if len(hand_positions) > 30:
            hand_positions.pop(0)
        
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame_show, (x, y), 3, (0, 255, 0), -1)
        
        cx = int(wrist.x * w)
        cy = int(wrist.y * h)
        cv2.circle(frame_show, (cx, cy), 15, (0, 255, 255), 2)
        cv2.putText(frame_show, "HAND", (cx - 25, cy - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cube_cx, cube_cy = w - 120, h // 2
    
    draw_3d_cube(frame_show, angles, cube_cx, cube_cy, scale=70)
    
    detected_colors = []
    grid_size = 50
    grid_positions = [
        (cube_cx - 80, cube_cy - 80),
        (cube_cx + 80, cube_cy - 80),
        (cube_cx - 80, cube_cy + 80),
        (cube_cx + 80, cube_cy + 80),
    ]
    
    for i, (gx, gy) in enumerate(grid_positions):
        color = detect_cube_color(frame, gx - 20, gy - 20, 40, 40)
        if color:
            detected_colors.append(color)
            color_bgr = CUBE_COLORS[color]
            cv2.rectangle(frame_show, (gx-20, gy-20), (gx+20, gy+20), color_bgr, -1)
            cv2.rectangle(frame_show, (gx-20, gy-20), (gx+20, gy+20), (255, 255, 255), 1)
            cv2.putText(frame_show, color[:3].upper(), (gx-18, gy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    dx, dy = detect_wave(hand_positions)
    
    current_time = time.time()
    
    if hand_detected and (abs(dx) > 0.01 or abs(dy) > 0.01):
        if current_time - last_wave_time > 0.1:
            angles[1] += dx
            angles[0] += dy
            last_wave_time = current_time
    
    angles[0] = max(-math.pi/2, min(math.pi/2, angles[0]))
    angles[1] = max(-math.pi, min(math.pi, angles[1]))
    
    key = cv2.waitKey(20) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord(' '):
        if current_time - last_reset_time > 0.5:
            angles = [0.0, 0.0, 0.0]
            hand_positions = []
            last_reset_time = current_time
            print("[重置] 魔方角度归零")
    
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "手势: 检测到" if hand_detected else "手势: 未检测"
    cv2.putText(frame_show, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    cv2.putText(frame_show, f"魔方角度 X:{math.degrees(angles[0]):.1f} Y:{math.degrees(angles[1]):.1f}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if detected_colors:
        cv2.putText(frame_show, f"检测颜色: {', '.join(detected_colors[:3])}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    cv2.rectangle(frame_show, (0, 0), (250, 110), (0, 0, 0), -1)
    cv2.rectangle(frame_show, (0, 0), (250, 110), (100, 100, 100), 1)
    cv2.putText(frame_show, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame_show, f"X:{math.degrees(angles[0]):.1f} Y:{math.degrees(angles[1]):.1f}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    if detected_colors:
        cv2.putText(frame_show, f"颜色: {', '.join(detected_colors[:3])}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    cv2.putText(frame_show, "右侧为3D魔方", (cube_cx - 50, cube_cy + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("挥手魔方控制", frame_show)

cap.release()
cv2.destroyAllWindows()
detector.close()
print("程序已退出")
