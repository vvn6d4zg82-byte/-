import os
import sys
import cv2
import serial
import time
import numpy as np
import torch

sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')

from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose

MODEL_PATH = r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main\weights\ReXNetV1-size-256-wingloss.pth'
IMG_SIZE = 256
NUM_CLASSES = 42

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

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请从百度网盘下载: https://pan.baidu.com/s/1Ur6Ikp31XGEuA3hQjYzwIw")
        print("密码: 99f3")
        return None
    
    model = ReXNetV1(num_classes=NUM_CLASSES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    chkpt = torch.load(model_path, map_location=device)
    model.load_state_dict(chkpt)
    print(f"模型加载成功: {model_path}")
    return model, device

def preprocess_image(img, bbox, img_size=256):
    x_min, y_min, x_max, y_max = bbox
    w = max(abs(x_max - x_min), abs(y_max - y_min))
    w = w * 1.1
    
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    
    x1 = int(x_mid - w / 2)
    y1 = int(y_mid - w / 2)
    x2 = int(x_mid + w / 2)
    y2 = int(y_mid + w / 2)
    
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    x2 = np.clip(x2, 0, img.shape[1] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)
    y2 = np.clip(y2, 0, img.shape[0] - 1)
    
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    crop = crop.astype(np.float32)
    crop = (crop - 128.) / 256.
    crop = crop.transpose(2, 0, 1)
    crop = torch.from_numpy(crop).unsqueeze_(0)
    
    return crop, x1, y1

def get_hand_keypoints(model, device, img, bbox):
    crop, offset_x, offset_y = preprocess_image(img, bbox, IMG_SIZE)
    
    if torch.cuda.is_available():
        crop = crop.cuda()
    
    with torch.no_grad():
        output = model(crop.float())
        output = output.cpu().numpy()
        output = np.squeeze(output)
    
    keypoints = {}
    img_h, img_w = img.shape[:2]
    for i in range(21):
        x = output[i * 2 + 0] * img_w + offset_x
        y = output[i * 2 + 1] * img_h + offset_y
        keypoints[str(i)] = {"x": x, "y": y}
    
    return keypoints

def detect_hand_bbox(frame):
    from mediapipe.tasks.python.vision import HandLandmarker
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
    
    base_options = base_options_module.BaseOptions(model_asset_path='hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    landmarker = HandLandmarker.create_from_options(options)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    
    landmarker.close()
    
    if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]
        x_coords = [lm.x * frame.shape[1] for lm in landmarks]
        y_coords = [lm.y * frame.shape[0] for lm in landmarks]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)
        
        return (x_min, y_min, x_max, y_max), landmarks
    
    return None, None

if __name__ == "__main__":
    result = load_model(MODEL_PATH)
    if result is None:
        print("模型加载失败，退出程序")
        sys.exit(1)
    
    model, device = result
    
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
        sys.exit(1)
    
    last_time = time.time()
    stable = 0
    last_hand = None
    
    print("=== handpose_x Hand Tracking with Servo Control ===")
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
        
        bbox, mp_landmarks = detect_hand_bbox(frame)
        
        if bbox is not None:
            keypoints = get_hand_keypoints(model, device, frame, bbox)
            
            hand_x = int(keypoints['9']['x'])
            hand_y = int(keypoints['9']['y'])
            
            if last_hand:
                dist = ((hand_x - last_hand[0])**2 + (hand_y - last_hand[1])**2)**0.5
                stable = min(stable + 1, 3) if dist < 20 else 0
            last_hand = (hand_x, hand_y)
            
            draw_bd_handpose(frame, keypoints, 0, 0)
            
            for i in range(21):
                x = int(keypoints[str(i)]['x'])
                y = int(keypoints[str(i)]['y'])
                cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
            
            fingertips = [keypoints['4'], keypoints['8'], keypoints['12'], keypoints['16'], keypoints['20']]
            palm_center = keypoints['9']
            
            distances = []
            for ft in fingertips:
                d = ((ft['x'] - palm_center['x'])**2 + (ft['y'] - palm_center['y'])**2)**0.5
                distances.append(d)
            
            avg_dist = sum(distances) / len(distances)
            is_fist = avg_dist < 0.12 * w
            
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
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
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
        
        cv2.imshow("handpose_x Hand Tracking", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    print("Done!")