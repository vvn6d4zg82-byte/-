import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import serial
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging

logging.getLogger('absl').setLevel(logging.ERROR)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

def skin_detect(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask

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
        print("Warning: No serial port found")
except:
    ser = None
    print("Warning: Serial port not available")

def move(s, a):
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"错误: 找不到模型文件 {model_path}")
    print("请从以下链接下载模型:")
    print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    min_hand_detection_confidence=0.05,
    min_hand_presence_confidence=0.05,
    min_tracking_confidence=0.05
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = frozenset([
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17)
])

def get_finger_state(landmarks):
    states = []
    tips = [4, 8, 12, 16, 20]
    bases = [2, 5, 9, 13, 17]
    
    thumb = landmarks[tips[0]]
    thumb_base = landmarks[bases[0]]
    states.append(1 if thumb.x > thumb_base.x else 0)
    
    for i in range(1, 5):
        tip = landmarks[tips[i]]
        base = landmarks[bases[i]]
        states.append(1 if tip.y < base.y else 0)
    
    return states

def recognize_gesture(finger_states):
    total = sum(finger_states)
    if finger_states == [0, 0, 0, 0, 0]:
        return "拳头", 0
    elif finger_states == [1, 1, 1, 1, 1]:
        return "手掌", 5
    elif finger_states == [0, 1, 0, 0, 0]:
        return "指物", 1
    elif finger_states == [0, 1, 1, 0, 0]:
        return "胜利", 2
    elif finger_states == [1, 0, 0, 0, 0]:
        return "点赞", 1
    return f"{total}手指", total

cap = cv2.VideoCapture(0)
for device_id in [0, 1]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None:
            print(f"Using camera device {device_id}")
            break
        cap.release()
else:
    print("ERROR: No camera available")
    exit(1)

cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

last_time = time.time()
stable = 0
last_gesture = None
last_finger_states = None
no_detection_count = 0
finger_states_buffer = []

print("=== MediaPipe 手势控制 ===")
print("按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    processed = preprocess_frame(frame)
    
    mask = skin_detect(frame)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)
    
    vis_frame = frame.copy()
    if results and results.hand_landmarks:
        no_detection_count = 0
        for landmarks in results.hand_landmarks:
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
            
            for edge in HAND_CONNECTIONS:
                p1 = landmarks[edge[0]]
                p2 = landmarks[edge[1]]
                cv2.line(vis_frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 0, 255), 1)
            
            finger_states = get_finger_state(landmarks)
            
            finger_states_buffer.append(tuple(finger_states))
            if len(finger_states_buffer) > 5:
                finger_states_buffer.pop(0)
            
            avg_state = [0]*5
            for fs in finger_states_buffer:
                for i in range(5):
                    avg_state[i] += fs[i]
            for i in range(5):
                avg_state[i] = 1 if avg_state[i] >= len(finger_states_buffer)/2 else 0
            
            gesture, fingers = recognize_gesture(avg_state)
            
            cx = int(landmarks[9].x * w)
            cy = int(landmarks[9].y * h)
            
            if last_gesture == gesture:
                stable = min(stable + 1, 3)
            else:
                stable = 0
            last_gesture = gesture
            last_finger_states = finger_states
            
            cv2.circle(vis_frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(vis_frame, (cx, cy), 14, (255, 255, 255), 2)
            
            cv2.putText(vis_frame, f"拇指:{avg_state[0]} 食指:{avg_state[1]} 中指:{avg_state[2]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"无名指:{avg_state[3]} 小指:{avg_state[4]}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"手势: {gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            s1 = 180 if avg_state[0] == 1 else 0
            s2 = 180 if avg_state[1] == 1 else 0
            s3 = 180 if avg_state[2] == 1 else 0
            s4 = 180 if avg_state[3] == 1 else 0
            s5 = 180 if avg_state[4] == 1 else 0
            
            cv2.putText(vis_frame, f"舵机1:{s1} 舵机2:{s2} 舵机3:{s3}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"舵机4:{s4} 舵机5:{s5}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if stable >= 1 and time.time() - last_time > 0.05:
                move(1, s1)
                move(2, s2)
                move(3, s3)
                move(4, s4)
                move(5, s5)
                last_time = time.time()
    else:
        no_detection_count += 1
        stable = 0
        if no_detection_count < 30 and last_finger_states is not None:
            finger_states = last_finger_states
            gesture = recognize_gesture(finger_states)[0]
            cv2.putText(vis_frame, f"手势: {gesture} (记忆)", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            last_gesture = None
            last_finger_states = None
    
    cv2.imshow("MediaPipe 手势控制", vis_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("程序结束!")