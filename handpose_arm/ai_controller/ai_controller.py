import os
import cv2
import numpy as np
import logging
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gesture_utils import recognize_gesture, get_palm_center


class AIController:
    def __init__(self, model_dir="models", use_skin_fallback=True):
        self.model_dir = model_dir
        self.use_skin_fallback = use_skin_fallback
        
        self.model = None
        self.scalers = None
        self.model_type = None
        self.model_loaded = False
        
        self.last_prediction = None
        self.prediction_count = 0
        
    def load_model(self, model_path=None):
        if model_path is None:
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_path = metadata.get('model_path')
        
        if model_path is None or not os.path.exists(model_path):
            logger.warning("没有找到已训练的模型，使用固定映射")
            self.model_loaded = False
            return False
        
        try:
            save_data = joblib.load(model_path)
            self.model = save_data['model']
            self.scalers = save_data['scalers']
            self.model_type = save_data['model_type']
            self.model_loaded = True
            
            logger.info(f"AI模型加载成功: {self.model_type}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model_loaded = False
            return False
    
    def keypoints_to_array(self, keypoints):
        arr = []
        for i in range(21):
            kp = keypoints.get(str(i), {"x": 0, "y": 0})
            arr.append(kp["x"])
            arr.append(kp["y"])
        return np.array(arr, dtype=np.float32)
    
    def predict_servos(self, keypoints):
        if self.model is None or self.scalers is None:
            return self._fixed_mapping(keypoints)
        
        try:
            X = self.keypoints_to_array(keypoints).reshape(1, -1)
            
            X = self.scalers['X'].transform(X)
            
            y = self.model.predict(X)
            
            y = self.scalers['y'].inverse_transform(y)
            
            result = {
                'base': int(np.clip(y[0][0], 0, 180)),
                'arm1': int(np.clip(y[0][1], 0, 180)),
                'arm2': int(np.clip(y[0][2], 0, 180)),
                'gripper': int(np.clip(y[0][3], 0, 180)),
                'rotation': int(np.clip(y[0][4], 0, 180))
            }
            
            self.last_prediction = result
            self.prediction_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return self._fixed_mapping(keypoints)
    
    def _fixed_mapping(self, keypoints):
        try:
            palm_center = get_palm_center(keypoints)
            hand_x = int(palm_center['x'])
            hand_y = int(palm_center['y'])
            
            gesture = recognize_gesture(keypoints, 640, 480)
            
            base = int(90 + (hand_x - 320) / 320 * 45)
            arm1 = int(90 + (hand_y - 240) / 240 * 45)
            arm2 = 90
            rotation = 90
            
            grip_angle = 90
            if gesture == " Fist":
                grip_angle = 30
            elif gesture == "Open":
                grip_angle = 150
            
            return {
                'base': int(np.clip(base, 0, 180)),
                'arm1': int(np.clip(arm1, 0, 180)),
                'arm2': int(np.clip(arm2, 0, 180)),
                'gripper': grip_angle,
                'rotation': int(np.clip(rotation, 0, 180))
            }
        except Exception as e:
            logger.error(f"固定映射失败: {e}")
            return {'base': 90, 'arm1': 90, 'arm2': 90, 'gripper': 90, 'rotation': 90}
    
    def predict_with_confidence(self, keypoints):
        result = self.predict_servos(keypoints)
        
        confidence = "high" if self.model_loaded else "low"
        
        return result, confidence
    
    def get_status(self):
        if self.model_loaded:
            return f"AI模式 ({self.model_type})"
        else:
            return "固定映射模式"


class AIRobotController:
    """Complete AI control system with camera integration"""
    
    def __init__(self):
        self.ai = AIController()
        self.arm = None
        self.running = False
        self.cap = None
        
        self.last_hand_pos = None
        self.stable_count = 0
        self.last_send_time = 0
        
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = 0
        
        import sys
        sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')
        
        self.detect_hand_bbox = None
        self.draw_handpose = None
        try:
            from hand_detector import detect_hand_bbox
            self.detect_hand_bbox = detect_hand_bbox
        except:
            pass
        try:
            from hand_data_iter.datasets import draw_bd_handpose
            self.draw_handpose = draw_bd_handpose
        except:
            pass
    
    def estimate_keypoints(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1)
        
        keypoints = {"0": {"x": cx, "y": cy}}
        
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0, 45, 90, 135, 180, 225, 270, 315, 0]
        radii = [0, 0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.98, 0.1, 0.25, 0.4, 0.55, 0.7, 0.82, 0.9, 0.95, 0.0]
        
        for i in range(21):
            angle_rad = angles[i] * 3.14159 / 180
            radius = size * radii[i]
            x = cx + radius * np.cos(angle_rad)
            y = cy + radius * np.sin(angle_rad)
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
    
    def connect_arm(self, ArmController=None):
        try:
            from arm_control import ArmController as AC
            self.arm = AC()
            if self.arm.connect():
                logger.info("机械臂连接成功")
                return True
        except Exception as e:
            logger.error(f"机械臂连接失败: {e}")
        return False
    
    def open_camera(self):
        for device_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(device_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        logger.info(f"摄像头打开: device={device_id}")
                        return True
                    cap.release()
            except Exception as e:
                logger.error(f"摄像头{device_id}打开失败: {e}")
        logger.error("无法打开摄像头")
        return False
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        if self.detect_hand_bbox:
            bbox = self.detect_hand_bbox(frame)
        else:
            bbox = None
        
        if bbox is None:
            self.stable_count = 0
            self.last_hand_pos = None
            return frame, None
        
        keypoints = self.estimate_keypoints(bbox, w, h)
        
        try:
            from gesture_utils import get_palm_center, recognize_gesture, get_grip_angle
            palm_center = get_palm_center(keypoints)
            hand_x = int(palm_center['x'])
            hand_y = int(palm_center['y'])
        except:
            hand_x, hand_y = w // 2, h // 2
        
        servo_output = self.ai.predict_with_confidence(keypoints)
        
        if isinstance(servo_output, tuple):
            servos, confidence = servo_output
        else:
            servos = servo_output
            confidence = "low"
        
        if self.draw_handpose:
            self.draw_handpose(frame, keypoints, 0, 0)
        
        for i in range(21):
            x = int(keypoints[str(i)]["x"])
            y = int(keypoints[str(i)]["y"])
            cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
        
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        cv2.putText(frame, f"S1:{servos['base']} S2:{servos['arm1']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S3:{servos['arm2']} S5:{servos['rotation']}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S4:{servos['gripper']}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        mode = self.ai.get_status()
        cv2.putText(frame, mode, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame, servo_values
    
    def run(self):
        print("\n=== AI机械臂控制系统 ===")
        
        self.ai.load_model()
        
        if not self.open_camera():
            return
        
        print("按 Q 退出")
        
        self.running = True
        
        import time
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame, servo_values = self.process_frame(frame)
            
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = current_time
            
            cv2.putText(frame, f"FPS: {self.fps}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("AI机械臂控制", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        self.stop()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.arm:
            self.arm.disconnect()
        cv2.destroyAllWindows()
        print("系统已停止")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='模型路径')
    args = parser.parse_args()
    
    ai = AIController()
    if args.model:
        ai.load_model(args.model)
    else:
        ai.load_model()
    
    controller = AIRobotController()
    controller.run()


if __name__ == "__main__":
    main()