import cv2
import serial
import time
import numpy as np
import urllib.request
import json
import requests

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

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    return np.degrees(angle)

def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    thumb_base = landmarks[2]
    index_base = landmarks[6]
    middle_base = landmarks[10]
    ring_base = landmarks[14]
    pinky_base = landmarks[18]
    
    wrist = landmarks[0]
    
    finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    finger_bases = [index_base, middle_base, ring_base, pinky_base]
    extended_fingers = []
    
    for tip, base in zip(finger_tips, finger_bases):
        dist_tip = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        dist_base = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2)**0.5
        if dist_tip > dist_base * 1.1:
            extended_fingers.append(1)
        else:
            extended_fingers.append(0)
    
    thumb_extended = 1 if ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)**0.5 > \
                          ((thumb_base.x - wrist.x)**2 + (thumb_base.y - wrist.y)**2)**0.5 * 1.1 else 0
    
    total_extended = sum(extended_fingers) + thumb_extended
    
    if total_extended == 0:
        return "fist"
    elif total_extended == 5:
        return "five"
    elif total_extended == 1 and extended_fingers[0] == 1:
        return "one"
    elif total_extended == 2 and extended_fingers[0] == 1 and extended_fingers[1] == 1:
        return "two"
    elif total_extended == 4:
        return "four"
    elif total_extended == 1 and thumb_extended == 1:
        return "thumbup"
    elif thumb_extended == 1 and sum(extended_fingers) == 0:
        return "ok"
    
    return "unknown"

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
            print(f"Using camera device {device_id}")
            break
        cap.release()
    else:
        print(f"Camera {device_id} not available")
        cap.release()
else:
    print("ERROR: No camera available")
    exit(1)

last_time = time.time()
alpha = 0.3
last_base, last_arm, last_arm2, last_grip, last_rot = 90, 90, 90, 90, 90

selection_start = None
selection_end = None
is_selecting = False

print("=== 手势控制 + 区域识别系统 ===")
print("手势: fist(握拳-夹爪闭合), five(张开-夹爪张开), one(食指-选择区域)")
print("区域选择: 伸出食指指向区域，按住不动选择区域")
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
    
    if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]
        
        gesture = recognize_gesture(landmarks)
        
        hand_x = int(landmarks[9].x * w)
        hand_y = int(landmarks[9].y * h)
        
        index_tip = landmarks[8]
        
        for finger in [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]:
            for i in range(len(finger) - 1):
                pt1 = (int(landmarks[finger[i]].x * w), int(landmarks[finger[i]].y * h))
                pt2 = (int(landmarks[finger[i+1]].x * w), int(landmarks[finger[i+1]].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        for lm in landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        
        cv2.circle(frame, (hand_x, hand_y), 10, (0, 0, 255), -1)
        
        if gesture == "one":
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(frame, index_pos, 15, (255, 0, 255), 2)
            
            if not is_selecting:
                selection_start = index_pos
                is_selecting = True
            selection_end = index_pos
            
            cv2.rectangle(frame, selection_start, selection_end, (255, 0, 255), 2)
            cv2.putText(frame, "选择区域中...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        elif is_selecting and selection_start and selection_end:
            x1 = min(selection_start[0], selection_end[0])
            y1 = min(selection_start[1], selection_end[1])
            x2 = max(selection_start[0], selection_end[0])
            y2 = max(selection_start[1], selection_end[1])
            
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                
                cv2.imwrite('selected_region.jpg', roi)
                
                try:
                    api_url = "https://api.ocr.space/parse/image"
                    with open('selected_region.jpg', 'rb') as f:
                        files = {'file': f}
                        data = {'language': 'chs', 'isOverlayRequired': 'true'}
                        response = requests.post(api_url, files=files, data=data, timeout=10)
                        result = response.json()
                        if result.get('ParsedResults'):
                            ocr_text = result['ParsedResults'][0]['ParsedText']
                            print(f"OCR识别结果: {ocr_text}")
                            cv2.putText(frame, f"OCR: {ocr_text[:30]}...", (10, h-100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                except Exception as e:
                    print(f"OCR识别失败: {e}")
                
                print(f"选择区域: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            selection_start = None
            selection_end = None
            is_selecting = False
        else:
            selection_start = None
            selection_end = None
            is_selecting = False
        
        is_fist = (gesture == "fist")
        
        grip = 180 if is_fist else 90
        gesture_cn = {"fist": "握拳", "five": "张开", "one": "选择", "two": "2", "four": "4", "thumbup": "点赞", "ok": "OK"}
        
        new_base = int(15 + 150 * hand_x / w)
        new_arm = int(15 + 150 * hand_y / h)
        new_arm2 = int(165 - 150 * hand_y / h)
        new_rot = int(180 - 180 * hand_x / w)
        
        smooth_base = int(last_base + alpha * (new_base - last_base))
        smooth_arm = int(last_arm + alpha * (new_arm - last_arm))
        smooth_arm2 = int(last_arm2 + alpha * (new_arm2 - last_arm2))
        smooth_grip = int(last_grip + alpha * (grip - last_grip))
        smooth_rot = int(last_rot + alpha * (new_rot - last_rot))
        
        last_base, last_arm, last_arm2, last_grip, last_rot = smooth_base, smooth_arm, smooth_arm2, smooth_grip, smooth_rot
        
        cv2.putText(frame, f"底座:{smooth_base} 大臂:{smooth_arm}", (10, h-80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"小臂:{smooth_arm2} 旋转:{smooth_rot}", (10, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"夹爪:{smooth_grip} 手势:{gesture_cn.get(gesture, gesture)}", (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        if time.time() - last_time > 0.06:
            move(1, smooth_base)
            move(2, smooth_arm)
            move(3, smooth_arm2)
            move(4, smooth_grip)
            move(5, smooth_rot)
            last_time = time.time()
    else:
        last_base, last_arm, last_arm2, last_grip, last_rot = 90, 90, 90, 90, 90
        selection_start = None
        selection_end = None
        is_selecting = False
    
    cv2.imshow("手势识别 + 区域选择", frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")