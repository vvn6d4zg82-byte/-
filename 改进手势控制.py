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

def recognize_gesture(landmarks):
    wrist = landmarks[0]
    palm_center = landmarks[9]
    
    finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    finger_bases = [landmarks[2], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
    
    distances_to_palm = []
    
    for tip in finger_tips:
        dist_tip_palm = ((tip.x - palm_center.x)**2 + (tip.y - palm_center.y)**2)**0.5
        distances_to_palm.append(dist_tip_palm)
    
    palm_radius = ((landmarks[0].x - palm_center.x)**2 + (landmarks[0].y - palm_center.y)**2)**0.5
    avg_dist_to_palm = sum(distances_to_palm) / len(distances_to_palm)
    
    if avg_dist_to_palm < palm_radius * 0.55:
        return "fist"
    
    extended_fingers = 0
    for tip, base in zip(finger_tips, finger_bases):
        dist_tip_wrist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        dist_base_wrist = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2)**0.5
        if dist_tip_wrist > dist_base_wrist * 1.1:
            extended_fingers += 1
    
    if extended_fingers == 0:
        return "fist"
    elif extended_fingers == 5:
        return "five"
    elif extended_fingers == 1 and distances_to_palm[1] > palm_radius * 0.5:
        return "point"
    elif extended_fingers == 2:
        return "peace"
    elif extended_fingers == 1 and distances_to_palm[0] > palm_radius * 0.5:
        return "thumb_up"
    
    return "unknown"

base_options = base_options_module.BaseOptions(model_asset_path='hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
landmarker = HandLandmarker.create_from_options(options)

for device_id in [0, 1]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None:
            print(f"Using camera device {device_id}")
            break
        cap.release()
    else:
        print(f"Camera {device_id} not available")
        cap.release()
else:
    print("ERROR: No camera available")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_time = time.time()
alpha = 0.5
last_base, last_arm, last_arm2, last_grip, last_rot = 90, 90, 90, 90, 90

hand_lost_count = 0
last_valid_pos = None
Grip_state = "open"
last_gesture = "unknown"

print("=== 改进版手势控制 ===")
print("手势: fist(握拳-闭合), five(张开), point(食指-选择)")
print("Q: quit")
print("")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, retrying...")
        time.sleep(0.5)
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    
    current_gesture = "unknown"
    
    if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
        hand_lost_count = 0
        landmarks = result.hand_landmarks[0]
        
        current_gesture = recognize_gesture(landmarks)
        last_gesture = current_gesture
        
        hand_x = int(landmarks[9].x * w)
        hand_y = int(landmarks[9].y * h)
        
        last_valid_pos = (hand_x, hand_y)
        
        for finger in [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]:
            for i in range(len(finger) - 1):
                pt1 = (int(landmarks[finger[i]].x * w), int(landmarks[finger[i]].y * h))
                pt2 = (int(landmarks[finger[i+1]].x * w), int(landmarks[finger[i+1]].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        for lm in landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)
        
        cv2.circle(frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
        
        gesture_color = (0, 255, 0) if current_gesture == "fist" else (255, 0, 255)
        
        if current_gesture == "fist":
            grip = 180
            Grip_state = "closed"
        else:
            grip = 90
            Grip_state = "open"
        
        new_base = int(30 + 120 * hand_x / w)
        new_arm = int(30 + 120 * hand_y / h)
        new_arm2 = int(150 - 120 * hand_y / h)
        new_rot = int(180 - 160 * hand_x / w)
        
        smooth_base = int(last_base + alpha * (new_base - last_base))
        smooth_arm = int(last_arm + alpha * (new_arm - last_arm))
        smooth_arm2 = int(last_arm2 + alpha * (new_arm2 - last_arm2))
        smooth_grip = int(last_grip + 0.7 * (grip - last_grip))
        smooth_rot = int(last_rot + alpha * (new_rot - last_rot))
        
        last_base, last_arm, last_arm2, last_grip, last_rot = smooth_base, smooth_arm, smooth_arm2, smooth_grip, smooth_rot
        
        cv2.putText(frame, f"底座:{smooth_base} 大臂:{smooth_arm}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"小臂:{smooth_arm2} 旋转:{smooth_rot}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"夹爪:{smooth_grip} 手势:{current_gesture}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
        
        if time.time() - last_time > 0.05:
            move(1, smooth_base)
            move(2, smooth_arm)
            move(3, smooth_arm2)
            move(4, smooth_grip)
            move(5, smooth_rot)
            last_time = time.time()
    else:
        hand_lost_count += 1
        
        if hand_lost_count < 10 and last_valid_pos is not None:
            hand_x, hand_y = last_valid_pos
            
            current_gesture = last_gesture
            
            if current_gesture == "fist":
                grip = 180
            else:
                grip = 90
            
            new_base = int(30 + 120 * hand_x / w)
            new_arm = int(30 + 120 * hand_y / h)
            new_arm2 = int(150 - 120 * hand_y / h)
            new_rot = int(180 - 160 * hand_x / w)
            
            smooth_base = int(last_base + alpha * (new_base - last_base))
            smooth_arm = int(last_arm + alpha * (new_arm - last_arm))
            smooth_arm2 = int(last_arm2 + alpha * (new_arm2 - last_arm2))
            smooth_grip = int(last_grip + 0.5 * (grip - last_grip))
            smooth_rot = int(last_rot + alpha * (new_rot - last_rot))
            
            last_base, last_arm, last_arm2, last_grip, last_rot = smooth_base, smooth_arm, smooth_arm2, smooth_grip, smooth_rot
            
            cv2.putText(frame, f"底座:{smooth_base} 大臂:{smooth_arm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"小臂:{smooth_arm2} 旋转:{smooth_rot}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"夹爪:{smooth_grip} 手势:{current_gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "手丢失 - 保持最后位置", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if time.time() - last_time > 0.05:
                move(1, smooth_base)
                move(2, smooth_arm)
                move(3, smooth_arm2)
                move(4, smooth_grip)
                move(5, smooth_rot)
                last_time = time.time()
        else:
            last_base, last_arm, last_arm2, last_grip, last_rot = 90, 90, 90, 90, 90
            last_valid_pos = None
            cv2.putText(frame, "未检测到手", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("手势控制", frame)
    
    key = cv2.waitKey(16) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")