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

def analyze_fingers(landmarks):
    wrist = landmarks[0]
    palm_center = landmarks[9]
    
    palm_radius = ((wrist.x - palm_center.x)**2 + (wrist.y - palm_center.y)**2)**0.5
    
    finger_names = ["拇指", "食指", "中指", "无名指", "小指"]
    finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    finger_mcps = [landmarks[2], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    
    finger_states = []
    
    for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcps)):
        dist_tip_palm = ((tip.x - palm_center.x)**2 + (tip.y - palm_center.y)**2)**0.5
        dist_tip_wrist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        dist_mcp_wrist = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)**0.5
        
        is_extended = dist_tip_wrist > dist_mcp_wrist * 1.05
        is_curled = dist_tip_palm < palm_radius * 0.6
        
        if is_curled:
            finger_states.append("弯曲")
        elif is_extended:
            finger_states.append("伸直")
        else:
            finger_states.append("半弯")
    
    avg_dist = sum([((tip.x - palm_center.x)**2 + (tip.y - palm_center.y)**2)**0.5 
                   for tip in finger_tips]) / 5
    is_fist = avg_dist < palm_radius * 0.55
    is_five = all(s == "伸直" for s in finger_states)
    
    return finger_states, is_fist, is_five

base_options = base_options_module.BaseOptions(model_asset_path='hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
    num_hands=2,
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

last_states = {}
last_base, last_arm, last_arm2, last_grip, last_rot = 90, 90, 90, 90, 90
left_last = {"base": 90, "arm": 90, "arm2": 90, "grip": 90, "rot": 90}
right_last = {"base": 90, "arm": 90, "arm2": 90, "grip": 90, "rot": 90}

print("=== 双手+手指捕捉 ===")
print("左手: 底座+大臂+小臂")
print("右手: 旋转+夹爪+手势状态")
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
    
    hands_info = []
    
    if result and result.hand_landmarks:
        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            hand_label = "左手" if hand_idx == 0 else "右手"
            
            finger_states, is_fist, is_five = analyze_fingers(landmarks)
            
            palm_center = landmarks[9]
            hand_x = int(palm_center.x * w)
            hand_y = int(palm_center.y * h)
            
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            
            for finger in [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]:
                for i in range(len(finger) - 1):
                    pt1 = (int(landmarks[finger[i]].x * w), int(landmarks[finger[i]].y * h))
                    pt2 = (int(landmarks[finger[i+1]].x * w), int(landmarks[finger[i+1]].y * h))
                    color = (0, 255, 0) if finger_states[finger[1]//4] == "伸直" else (0, 0, 255)
                    cv2.line(frame, pt1, pt2, color, 2)
            
            for i, lm in enumerate(landmarks):
                pt = (int(lm.x * w), int(lm.y * h))
                color = (0, 255, 255)
                if i in [4, 8, 12, 16, 20]:
                    color = (255, 0, 255)
                cv2.circle(frame, pt, 4, color, -1)
            
            cv2.circle(frame, (hand_x, hand_y), 12, (0, 0, 255), -1)
            
            y_offset = 30 if hand_idx == 0 else 120
            
            status_text = "握拳" if is_fist else ("张开" if is_five else "未知")
            if hand_idx == 0:
                new_base = int(30 + 120 * hand_x / w)
                new_arm = int(30 + 120 * hand_y / h)
                new_arm2 = int(150 - 120 * hand_y / h)
                
                smooth_base = int(left_last["base"] + alpha * (new_base - left_last["base"]))
                smooth_arm = int(left_last["arm"] + alpha * (new_arm - left_last["arm"]))
                smooth_arm2 = int(left_last["arm2"] + alpha * (new_arm2 - left_last["arm2"]))
                
                left_last["base"], left_last["arm"], left_last["arm2"] = smooth_base, smooth_arm, smooth_arm2
                
                cv2.putText(frame, f"左手 底座:{smooth_base} 大臂:{smooth_arm}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"左手 小臂:{smooth_arm2} 状态:{status_text}", (10, y_offset+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                cv2.putText(frame, f"手指:{finger_states[0]}{finger_states[1]}{finger_states[2]}{finger_states[3]}{finger_states[4]}", 
                        (10, y_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                new_rot = int(180 - 160 * hand_x / w)
                grip = 180 if is_fist else 90
                
                smooth_rot = int(right_last["rot"] + alpha * (new_rot - right_last["rot"]))
                smooth_grip = int(right_last["grip"] + 0.7 * (grip - right_last["grip"]))
                
                right_last["rot"], right_last["grip"] = smooth_rot, smooth_grip
                
                cv2.putText(frame, f"右手 旋转:{smooth_rot}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"右手 夹爪:{smooth_grip} 状态:{status_text}", (10, y_offset+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                cv2.putText(frame, f"手指:{finger_states[0]}{finger_states[1]}{finger_states[2]}{finger_states[3]}{finger_states[4]}", 
                        (10, y_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            hands_info.append({
                "hand": hand_label,
                "x": hand_x,
                "y": hand_y,
                "fingers": finger_states,
                "is_fist": is_fist,
                "is_five": is_five
            })
    
    if len(hands_info) >= 1:
        if time.time() - last_time > 0.05:
            if len(hands_info) >= 1:
                move(1, left_last["base"])
                move(2, left_last["arm"])
                move(3, left_last["arm2"])
            if len(hands_info) >= 2:
                move(4, right_last["grip"])
                move(5, right_last["rot"])
            last_time = time.time()
    elif len(hands_info) == 0:
        left_last = {"base": 90, "arm": 90, "arm2": 90, "grip": 90, "rot": 90}
        right_last = {"base": 90, "arm": 90, "arm2": 90, "grip": 90, "rot": 90}
        cv2.putText(frame, "未检测到手", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("双手+手指捕捉", frame)
    
    key = cv2.waitKey(16) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")