import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import serial
import time
import math

try:
    for port in ['COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM3', 'COM4']:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Connected to {port}")
            time.sleep(2)
            break
        except:
            continue
    else:
        ser = None
except:
    ser = None

def move(s, a):
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass

def rotate_point(x, y, z, angle, axis):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    if axis == 'x':
        y_new = y * cos_a - z * sin_a
        z_new = y * sin_a + z * cos_a
        return x, y_new, z_new
    elif axis == 'y':
        x_new = x * cos_a + z * sin_a
        z_new = -x * sin_a + z * cos_a
        return x_new, y, z_new
    else:
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        return x_new, y_new, z

def project_3d(x, y, z, cx, cy, scale):
    z += 3
    fov = 500
    factor = fov / (fov + z * 50)
    px = int(cx + x * scale * factor)
    py = int(cy + y * scale * factor)
    return px, py

def draw_cube(frame, angles, cx, cy):
    scale = 80
    
    vertices = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vx, vy, vz = x, y, z
                vx, vy, vz = rotate_point(vx, vy, vz, angles[0], 'x')
                vx, vy, vz = rotate_point(vx, vy, vz, angles[1], 'y')
                vx, vy, vz = rotate_point(vx, vy, vz, angles[2], 'z')
                vertices.append((vx, vy, vz))
    
    faces = [
        ([0,1,3,2], (0, 255, 255)),
        ([4,5,7,6], (255, 0, 255)),
        ([0,4,6,2], (255, 255, 0)),
        ([1,5,7,3], (0, 255, 0)),
        ([0,1,5,4], (0, 0, 255)),
        ([2,3,7,6], (255, 0, 0))
    ]
    
    projected = []
    for v in vertices:
        px, py = project_3d(v[0], v[1], v[2], cx, cy, scale)
        projected.append((px, py))
    
    for face, color in faces:
        pts = np.array([projected[i] for i in face], np.int32)
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (255,255,255), 2)
    
    for v in projected:
        cv2.circle(frame, v, 5, (255,255,255), -1)
    
    return projected

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as image_module

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

angles = [0.0, 0.0, 0.0]
angle_speeds = [0.0, 0.0, 0.0]

servo_values = [90, 90, 90, 90, 90]
target_values = [90, 90, 90, 90, 90]

last_time = time.time()

hand_history_x = []
hand_history_y = []

print("=== MediaPipe Hand Control ===")
print("Move hand to rotate cube")
print("Key: Space=reset Q=quit")

cv2.namedWindow("Cube Control")

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
    
    dx, dy = 0, 0
    hand_detected = False
    
    if results and results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        hand_detected = True
        
        wrist = landmarks[0]
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        
        hand_history_x.append(wrist_x)
        hand_history_y.append(wrist_y)
        
        if len(hand_history_x) >= 2:
            dx = (hand_history_x[-1] - hand_history_x[0]) / w
            dy = (hand_history_y[-1] - hand_history_y[0]) / h
        
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame_show, (x, y), 1, (0, 255, 0), 1)
    
    if abs(dx) > 0.01:
        angle_speeds[1] = dx * 0.5
    else:
        angle_speeds[1] = 0
    
    if abs(dy) > 0.01:
        angle_speeds[0] = dy * 0.5
    else:
        angle_speeds[0] = 0
    
    key = cv2.waitKey(20) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord(' '):
        target_values = [90, 90, 90, 90, 90]
        angle_speeds = [0.0, 0.0, 0.0]
        angles = [0.0, 0.0, 0.0]
        hand_history_x = []
        hand_history_y = []
    
    for i in range(3):
        angles[i] += angle_speeds[i]
        angles[i] = max(-3.14, min(3.14, angles[i]))
    
    for i in range(5):
        if abs(servo_values[i] - target_values[i]) > 1:
            step = 3 if servo_values[i] < target_values[i] else -3
            servo_values[i] += step
    
    draw_cube(frame_show, angles, 540, 240)
    
    if hand_detected:
        cv2.putText(frame_show, "Hand: ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame_show, "Hand: OFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.putText(frame_show, f"S1:{servo_values[0]} S2:{servo_values[1]} S3:{servo_values[2]}", (10, 380),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame_show, f"S4:{servo_values[3]} S5:{servo_values[4]}", (10, 410),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if time.time() - last_time > 0.05:
        for i in range(5):
            move(i+1, servo_values[i])
        last_time = time.time()
    
    cv2.imshow("Cube Control", frame_show)

cap.release()
cv2.destroyAllWindows()
detector.close()
if ser:
    ser.close()
print("Done!")