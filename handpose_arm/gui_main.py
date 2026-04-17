import sys
import os
import cv2
import numpy as np
import torch
import time
import threading
import logging
from collections import deque
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')

try:
    from models.rexnetv1 import ReXNetV1
    from hand_data_iter.datasets import draw_bd_handpose
except Exception as e:
    logger.error(f"导入handpose_x模型失败: {e}")
    ReXNetV1 = None
    draw_bd_handpose = None

try:
    from hand_detector import detect_hand_bbox, get_skin_mask
except Exception as e:
    logger.error(f"导入hand_detector失败: {e}")
    detect_hand_bbox = None
    get_skin_mask = None

try:
    from gesture_utils import (
        recognize_gesture,
        map_position_to_servo,
        get_grip_angle,
        get_palm_center
    )
except Exception as e:
    logger.error(f"导入gesture_utils失败: {e}")
    recognize_gesture = None
    map_position_to_servo = None
    get_grip_angle = None
    get_palm_center = None

try:
    from arm_control import ArmController
except Exception as e:
    logger.error(f"导入arm_control失败: {e}")
    ArmController = None

MODEL_PATH = r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main\weights\ReXNetV1-size-256-wingloss.pth'
IMG_SIZE = 256
NUM_CLASSES = 42
MIN_ARM_CONNECT_INTERVAL = 0.08
ERROR_LOG_MAX = 100


class ErrorLogger:
    def __init__(self, max_size=ERROR_LOG_MAX):
        self.errors = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_error(self, error_type, message, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_entry = {
            "timestamp": timestamp,
            "type": error_type,
            "message": message,
            "details": details
        }
        with self.lock:
            self.errors.append(error_entry)
        logger.error(f"[{error_type}] {message} - {details}")
        return error_entry
    
    def get_errors(self):
        with self.lock:
            return list(self.errors)
    
    def clear(self):
        with self.lock:
            self.errors.clear()
    
    def get_recent_errors(self, count=10):
        with self.lock:
            return list(self.errors)[-count:]


class HandPoseArmGUI:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model = None
        self.device = None
        self.arm = None
        self.error_logger = ErrorLogger()
        
        self.use_model = False
        self.running = False
        self.cap = None
        self.current_frame = None
        self.keypoints = None
        self.servo_values = None
        
        self.last_hand_pos = None
        self.stable_count = 0
        self.last_send_time = 0
        
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                self.error_logger.add_error("MODEL", "模型文件不存在", self.model_path)
                return False
            
            if ReXNetV1 is None:
                self.error_logger.add_error("IMPORT", "ReXNetV1模块未导入")
                return False
            
            self.model = ReXNetV1(num_classes=NUM_CLASSES)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            chkpt = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(chkpt)
            self.use_model = True
            self.error_logger.add_error("INFO", "HandPose-X模型加载成功", self.model_path)
            return True
        except Exception as e:
            self.error_logger.add_error("MODEL", f"加载HandPose-X模型失败: {e}")
            return False
    
    def load_arm(self):
        try:
            if ArmController is None:
                self.error_logger.add_error("IMPORT", "ArmController模块未导入")
                return False
            
            self.arm = ArmController()
            if self.arm.connect():
                self.error_logger.add_error("INFO", "机械臂连接成功")
                return True
            else:
                self.error_logger.add_error("ARM", "无法连接机械臂")
                return False
        except Exception as e:
            self.error_logger.add_error("ARM", f"机械臂连接失败: {e}")
            return False
    
    def preprocess_image(self, img, bbox, img_size=IMG_SIZE):
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
        if self.model is None:
            return None
        
        try:
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
        except Exception as e:
            self.error_logger.add_error("PREDICT", f"关键点预测失败: {e}")
            return None
    
    def estimate_keypoints(self, bbox, frame_w, frame_h):
        try:
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
        except Exception as e:
            self.error_logger.add_error("ESTIMATE", f"估算关键点失败: {e}")
            return None
    
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
        try:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            if detect_hand_bbox is None:
                self.error_logger.add_error("IMPORT", "detect_hand_bbox未导入")
                return frame, None, None
            
            bbox = detect_hand_bbox(frame)
            if bbox is None:
                self.stable_count = 0
                self.last_hand_pos = None
                return frame, None, None
            
            if self.use_model and self.model is not None:
                keypoints = self.predict_keypoints(frame, bbox)
                if keypoints is None:
                    keypoints = self.estimate_keypoints(bbox, w, h)
            else:
                keypoints = self.estimate_keypoints(bbox, w, h)
            
            if keypoints is None:
                return frame, None, None
            
            if get_palm_center is None or recognize_gesture is None or map_position_to_servo is None or get_grip_angle is None:
                self.error_logger.add_error("IMPORT", "gesture_utils函数未导入")
                return frame, None, None
            
            palm_center = get_palm_center(keypoints)
            hand_x = int(palm_center['x'])
            hand_y = int(palm_center['y'])
            
            gesture = recognize_gesture(keypoints, w, h)
            grip_angle = get_grip_angle(keypoints, w, h)
            
            base, arm1, arm2, rotation = map_position_to_servo(hand_x, hand_y, w, h)
            
            if self.use_model and draw_bd_handpose is not None:
                draw_bd_handpose(frame, keypoints, 0, 0)
            
            for i in range(21):
                x = int(keypoints[str(i)]['x'])
                y = int(keypoints[str(i)]['y'])
                cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            cv2.putText(frame, f"S1:{base} S2:{arm1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S3:{arm2} S5:{rotation}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S4:{grip_angle} {gesture}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            mode_text = "[HandPose-X]" if self.use_model else "[Skin Detect]"
            cv2.putText(frame, mode_text, (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            servo_values = {
                'base': base,
                'arm1': arm1,
                'arm2': arm2,
                'gripper': grip_angle,
                'rotation': rotation,
                'gesture': gesture
            }
            
            if self.arm and self.arm.is_connected():
                current_time = time.time()
                if self.is_hand_stable(servo_values['base'], servo_values['arm1']):
                    if current_time - self.last_send_time > MIN_ARM_CONNECT_INTERVAL:
                        try:
                            self.arm.set_all_servos(
                                servo_values['base'],
                                servo_values['arm1'],
                                servo_values['arm2'],
                                servo_values['gripper'],
                                servo_values['rotation']
                            )
                            self.last_send_time = current_time
                        except Exception as e:
                            self.error_logger.add_error("ARM", f"发送舵机指令失败: {e}")
            
            return frame, keypoints, servo_values
        except Exception as e:
            self.error_logger.add_error("PROCESS", f"处理帧失败: {e}")
            return frame, None, None
    
    def open_camera(self):
        for device_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.error_logger.add_error("INFO", f"摄像头打开成功 device={device_id}")
                        return True
                    cap.release()
            except Exception as e:
                self.error_logger.add_error("CAMERA", f"打开摄像头{device_id}失败: {e}")
        
        self.error_logger.add_error("CAMERA", "无法打开任何摄像头")
        return False
    
    def close_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def start(self):
        self.load_model()
        self.load_arm()
        
        if not self.open_camera():
            return False
        
        self.running = True
        self.error_logger.add_error("INFO", "系统启动成功")
        return True
    
    def stop(self):
        self.running = False
        self.close_camera()
        if self.arm:
            self.arm.disconnect()
        self.error_logger.add_error("INFO", "系统已停止")


def main():
    app = HandPoseArmGUI()
    
    if app.start():
        print("系统启动成功!")
        print("按Q退出")
        
        while app.running:
            if app.cap and app.cap.isOpened():
                ret, frame = app.cap.read()
                if ret:
                    frame, keypoints, servo_values = app.process_frame(frame)
                    cv2.imshow("HandPose Arm Control", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
    
    app.stop()
    cv2.destroyAllWindows()
    
    errors = app.error_logger.get_errors()
    print(f"\n错误日志 (共{len(errors)}条):")
    for err in errors[-10:]:
        print(f"  [{err['timestamp']}] [{err['type']}] {err['message']} - {err['details']}")


if __name__ == "__main__":
    main()