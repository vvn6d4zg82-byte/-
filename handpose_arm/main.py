import os
import sys
import cv2
import numpy as np
import torch
import time
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')

from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose

from hand_detector import detect_hand_bbox
from gesture_utils import (
    recognize_gesture,
    map_position_to_servo,
    get_grip_angle,
    get_palm_center
)
from arm_control import ArmController

MODEL_PATH = r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main\weights\ReXNetV1-size-256-wingloss.pth'
MEDIAPIPE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hand_landmarker.task')
IMG_SIZE = 256
NUM_CLASSES = 42
MIN_ARM_CONNECT_INTERVAL = 0.08


class HandPoseArmController:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.arm = ArmController()
        self.use_mediapipe = False
        self.mp_landmarker = None
        
        self.last_hand_pos = None
        self.stable_count = 0
        self.last_send_time = 0
        
    def load_handpose_x_model(self):
        if not os.path.exists(self.model_path):
            print(f"HandPose X model not found: {self.model_path}")
            return False
        
        try:
            self.model = ReXNetV1(num_classes=NUM_CLASSES)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            chkpt = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(chkpt)
            print(f"HandPose X model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading HandPose X model: {e}")
            return False
    
    def load_skin_detector(self):
        try:
            import cv2
            import numpy as np
            print(f"Skin detection hand detector ready")
            return True
        except Exception as e:
            print(f"Error loading skin detector: {e}")
            return False
    
    def load_model(self):
        if self.load_skin_detector():
            return True
        
        if self.load_handpose_x_model():
            return True
        
        return False
    
    def get_hand_bbox(self, frame):
        bbox = detect_hand_bbox(frame)
        return bbox
    
    def mediapipe_to_keypoints(self, mp_landmarks, offset_x, offset_y, frame_shape):
        keypoints = {}
        for i, lm in enumerate(mp_landmarks):
            x = lm.x * frame_shape[1]
            y = lm.y * frame_shape[0]
            keypoints[str(i)] = {"x": x, "y": y}
        return keypoints
    
    def estimate_keypoints_from_bbox(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1)
        
        keypoints = {}
        keypoints["0"] = {"x": cx, "y": cy}
        
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0, 45, 90, 135, 180, 225, 270, 315, 0]
        radii = [0, 0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.98, 0.1, 0.25, 0.4, 0.55, 0.7, 0.82, 0.9, 0.95, 0.0]
        
        for i in range(21):
            angle_rad = angles[i] * 3.14159 / 180
            radius = size * radii[i]
            x = cx + radius * np.cos(angle_rad)
            y = cy + radius * np.sin(angle_rad)
            keypoints[str(i)] = {"x": x, "y": y}
        
        return keypoints
    
    def preprocess_image(self, img, bbox, img_size=256):
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
    
    def predict_keypoints(self, img, bbox):
        crop, offset_x, offset_y = self.preprocess_image(img, bbox, IMG_SIZE)
        
        if torch.cuda.is_available():
            crop = crop.cuda()
        
        with torch.no_grad():
            output = self.model(crop.float())
            output = output.cpu().numpy()
            output = np.squeeze(output)
        
        keypoints = {}
        img_h, img_w = img.shape[:2]
        for i in range(21):
            x = output[i * 2 + 0] * img_w + offset_x
            y = output[i * 2 + 1] * img_h + offset_y
            keypoints[str(i)] = {"x": x, "y": y}
        
        return keypoints
    
    def is_hand_stable(self, hand_x, hand_y):
        if self.last_hand_pos is None:
            self.last_hand_pos = (hand_x, hand_y)
            return False
        
        dist = ((hand_x - self.last_hand_pos[0])**2 + (hand_y - self.last_hand_pos[1])**2)**0.5
        
        if dist < 20:
            self.stable_count = min(self.stable_count + 1, 3)
        else:
            self.stable_count = 0
        
        self.last_hand_pos = (hand_x, hand_y)
        return self.stable_count >= 2
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        if self.model is not None:
            bbox = detect_hand_bbox(frame)
            if bbox is None:
                self.stable_count = 0
                self.last_hand_pos = None
                return frame, None, None
            
            keypoints = self.predict_keypoints(frame, bbox)
        else:
            bbox = self.get_hand_bbox(frame)
            if bbox is None:
                self.stable_count = 0
                self.last_hand_pos = None
                return frame, None, None
            
            keypoints = self.estimate_keypoints_from_bbox(bbox, w, h)
        
        palm_center = get_palm_center(keypoints)
        hand_x = int(palm_center['x'])
        hand_y = int(palm_center['y'])
        
        gesture = recognize_gesture(keypoints, w, h)
        grip_angle = get_grip_angle(keypoints, w, h)
        
        base, arm1, arm2, rotation = map_position_to_servo(hand_x, hand_y, w, h)
        
        if self.model is not None:
            draw_bd_handpose(frame, keypoints, 0, 0)
            for i in range(21):
                x = int(keypoints[str(i)]['x'])
                y = int(keypoints[str(i)]['y'])
                cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
        else:
            for i in range(21):
                x = int(keypoints[str(i)]['x'])
                y = int(keypoints[str(i)]['y'])
                cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
            
            connections = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
                          (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
                          (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)]
            colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
            for i, (p1, p2) in enumerate(connections):
                c = colors[i//4] if i < 20 else colors[0]
                pt1 = (int(keypoints[str(p1)]['x']), int(keypoints[str(p1)]['y']))
                pt2 = (int(keypoints[str(p2)]['x']), int(keypoints[str(p2)]['y']))
                cv2.line(frame, pt1, pt2, c, 2)
        
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        cv2.putText(frame, f"S1:{base} S2:{arm1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S3:{arm2} S5:{rotation}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S4:{grip_angle} {gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if self.model is not None:
            cv2.putText(frame, "[HandPose-X]", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "[Skin Detect]", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        servo_values = {
            'base': base,
            'arm1': arm1,
            'arm2': arm2,
            'gripper': grip_angle,
            'rotation': rotation,
            'gesture': gesture
        }
        
        return frame, keypoints, servo_values
    
    def run(self):
        if not self.load_model():
            print("Failed to load any model, exiting")
            sys.exit(1)
        
        self.arm.connect()
        
        cap = None
        for device_id in [0, 1]:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Using camera device {device_id}")
                    break
                cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            print("ERROR: No camera available")
            sys.exit(1)
        
        print("=== HandPose X + Robotic Arm Control ===")
        print("Q: quit")
        print("")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error, retrying...")
                continue
            
            frame, keypoints, servo_values = self.process_frame(frame)
            
            if servo_values is not None:
                current_time = time.time()
                
                if self.is_hand_stable(servo_values['base'], servo_values['arm1']):
                    if current_time - self.last_send_time > MIN_ARM_CONNECT_INTERVAL:
                        self.arm.set_all_servos(
                            servo_values['base'],
                            servo_values['arm1'],
                            servo_values['arm2'],
                            servo_values['gripper'],
                            servo_values['rotation']
                        )
                        self.last_send_time = current_time
            
            cv2.imshow("HandPose X - Robotic Arm Control", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.arm.disconnect()
        print("Done!")

if __name__ == "__main__":
    controller = HandPoseArmController()
    controller.run()