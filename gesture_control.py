import cv2
import serial
import time
import numpy as np

from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode

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

base_options = base_options_module.BaseOptions(model_asset_path='hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

for device_id in [0, 1]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None:
            print(f"Using camera device {device_id}, resolution: {frame_test.shape}")
            break
        else:
            print(f"Camera {device_id} opened but cannot read, trying next...")
            cap.release()
    else:
        print(f"Camera {device_id} not available")
        cap.release()
else:
    print("ERROR: No camera available")
    exit(1)

last_time = time.time()
stable = 0
last_hand = None

print("=== MediaPipe Hand Tracking with Servo Control ===")
print("Q: quit")
print("")

camera_retry_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        camera_retry_count += 1
        if camera_retry_count < 10:
            print(f"Camera error (retry {camera_retry_count}/10), retrying...")
            time.sleep(0.5)
            continue
        else:
            print("Camera failed multiple times, exiting")
            break
    
    camera_retry_count = 0
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    
    if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]
        
        hand_x = int(landmarks[9].x * w)
        hand_y = int(landmarks[9].y * h)
        
        if last_hand:
            dist = ((hand_x - last_hand[0])**2 + (hand_y - last_hand[1])**2)**0.5
            stable = min(stable + 1, 3) if dist < 20 else 0
        last_hand = (hand_x, hand_y)
        
        for finger in [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]:
            for i in range(len(finger) - 1):
                pt1 = (int(landmarks[finger[i]].x * w), int(landmarks[finger[i]].y * h))
                pt2 = (int(landmarks[finger[i+1]].x * w), int(landmarks[finger[i+1]].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        for lm in landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        
        cv2.circle(frame, (hand_x, hand_y), 10, (0, 0, 255), -1)
        cv2.circle(frame, (hand_x, hand_y), 15, (255, 255, 255), 2)
        
        fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        palm_center = landmarks[9]
        
        distances = []
        for ft in fingertips:
            d = ((ft.x - palm_center.x)**2 + (ft.y - palm_center.y)**2)**0.5
            distances.append(d)
        
        avg_dist = sum(distances) / len(distances)
        is_fist = avg_dist < 0.12
        
        grip = 180 if is_fist else 90
        gesture = "fist" if is_fist else "open"
        
        new_base = int(15 + 150 * hand_x / w)
        new_arm = int(15 + 150 * hand_y / h)
        new_arm2 = int(165 - 150 * hand_y / h)
        new_rot = int(180 - 180 * hand_x / w)
        
        cv2.putText(frame, f"S1:{new_base} S2:{new_arm}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S3:{new_arm2} S5:{new_rot}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S4:{grip} {gesture}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        if stable >= 2 and time.time() - last_time > 0.08:
            move(1, new_base)
            move(2, new_arm)
            move(3, new_arm2)
            move(4, grip)
            move(5, new_rot)
            last_time = time.time()
    else:
        stable = 0
        last_hand = None
    
    cv2.imshow("MediaPipe Hand Tracking", frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")