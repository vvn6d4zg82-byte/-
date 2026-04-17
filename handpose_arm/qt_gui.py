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

sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')

ReXNetV1 = None
draw_bd_handpose = None
detect_hand_bbox = None
get_skin_mask = None
recognize_gesture = None
map_position_to_servo = None
get_grip_angle = None
get_palm_center = None
ArmController = None

try:
    from models.rexnetv1 import ReXNetV1
    from hand_data_iter.datasets import draw_bd_handpose
except Exception as e:
    print(f"导入handpose_x模型失败: {e}")

try:
    from hand_detector import detect_hand_bbox, get_skin_mask
except Exception as e:
    print(f"导入hand_detector失败: {e}")

try:
    from gesture_utils import (
        recognize_gesture,
        map_position_to_servo,
        get_grip_angle,
        get_palm_center
    )
except Exception as e:
    print(f"导入gesture_utils失败: {e}")

try:
    from arm_control import ArmController
except Exception as e:
    print(f"导入arm_control失败: {e}")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QGroupBox, QFrame,
    QSpinBox, QComboBox, QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

MODEL_PATH = r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main\weights\ReXNetV1-size-256-wingloss.pth'
IMG_SIZE = 256
NUM_CLASSES = 42
MIN_ARM_CONNECT_INTERVAL = 0.08


class ErrorLogger:
    def __init__(self, max_size=100):
        self.errors = deque(maxlen=max_size)
    
    def add_error(self, error_type, message, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_entry = {
            "timestamp": timestamp,
            "type": error_type,
            "message": message,
            "details": details
        }
        self.errors.append(error_entry)
        return error_entry
    
    def get_errors(self):
        return list(self.errors)
    
    def clear(self):
        self.errors.clear()


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.running = True
    
    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            time.sleep(0.033)
    
    def stop(self):
        self.running = False


class HandPoseArmGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = MODEL_PATH
        self.model = None
        self.device = None
        self.arm = None
        self.error_logger = ErrorLogger()
        
        self.use_model = False
        self.running = False
        self.cap = None
        self.video_thread = None
        self.current_frame = None
        self.keypoints = None
        self.servo_values = None
        
        self.last_hand_pos = None
        self.stable_count = 0
        self.last_send_time = 0
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        self.setWindowTitle("手势控制机械臂系统")
        self.setGeometry(100, 100, 1280, 720)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        left_panel = self.create_video_panel()
        main_layout.addWidget(left_panel, 3)
        
        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_video_panel(self):
        group = QGroupBox("视频显示")
        layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.fps_label)
        
        group.setLayout(layout)
        return group
    
    def create_control_panel(self):
        group = QGroupBox("控制面板")
        layout = QVBoxLayout()
        
        btn_style = "QPushButton { padding: 10px; font-size: 14px; }"
        
        self.start_btn = QPushButton("启动系统")
        self.start_btn.setStyleSheet(btn_style)
        self.start_btn.clicked.connect(self.start_system)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止系统")
        self.stop_btn.setStyleSheet(btn_style)
        self.stop_btn.clicked.connect(self.stop_system)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        layout.addWidget(self.create_servo_panel())
        
        layout.addWidget(self.create_error_panel())
        
        group.setLayout(layout)
        return group
    
    def create_servo_panel(self):
        group = QGroupBox("舵机状态")
        layout = QVBoxLayout()
        
        servo_labels = [
            ("S1 (基座)", "base"),
            ("S2 (大臂)", "arm1"),
            ("S3 (小臂)", "arm2"),
            ("S4 (夹爪)", "gripper"),
            ("S5 (旋转)", "rotation"),
        ]
        
        self.servo_labels = {}
        for label_text, key in servo_labels:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(80)
            val_lbl = QLabel("---")
            val_lbl.setStyleSheet("font-weight: bold; color: #00AA00;")
            row.addWidget(lbl)
            row.addWidget(val_lbl)
            layout.addLayout(row)
            self.servo_labels[key] = val_lbl
        
        self.gesture_label = QLabel("手势: ---")
        self.gesture_label.setStyleSheet("font-weight: bold; color: #FF00FF;")
        layout.addWidget(self.gesture_label)
        
        group.setLayout(layout)
        return group
    
    def create_error_panel(self):
        group = QGroupBox("错误日志")
        layout = QVBoxLayout()
        
        self.error_table = QTableWidget()
        self.error_table.setColumnCount(4)
        self.error_table.setHorizontalHeaderLabels(["时间", "类型", "消息", "详情"])
        self.error_table.setMinimumHeight(200)
        layout.addWidget(self.error_table)
        
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_errors)
        btn_layout.addWidget(clear_btn)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_errors)
        btn_layout.addWidget(refresh_btn)
        
        layout.addLayout(btn_layout)
        
        group.setLayout(layout)
        return group
    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                self.error_logger.add_error("MODEL", "模型文件不存在", self.model_path)
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
            self.error_logger.add_error("MODEL", f"加载模型失败: {e}")
            self.use_model = False
            return False
    
    def load_arm(self):
        try:
            self.arm = ArmController()
            if self.arm.connect():
                self.error_logger.add_error("INFO", "机械臂连接成功")
                return True
            else:
                self.error_logger.add_error("ARM", "无法连接机械臂")
                return False
        except Exception as e:
            self.error_logger.add_error("ARM", f"连接失败: {e}")
            return False
    
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
    
    def process_frame(self, frame):
        try:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            bbox = detect_hand_bbox(frame) if detect_hand_bbox else None
            if bbox is None:
                self.stable_count = 0
                self.last_hand_pos = None
                return frame, None, None
            
            keypoints = self.estimate_keypoints(bbox, w, h)
            
            try:
                palm_center = get_palm_center(keypoints)
                hand_x = int(palm_center['x'])
                hand_y = int(palm_center['y'])
                
                gesture = recognize_gesture(keypoints, w, h)
                grip_angle = get_grip_angle(keypoints, w, h)
                
                base, arm1, arm2, rotation = map_position_to_servo(hand_x, hand_y, w, h)
            except Exception as e:
                self.error_logger.add_error("GESTURE", f"处理手势失败: {e}")
                base, arm1, arm2, rotation, gesture, grip_angle = 90, 90, 90, 90, "未知", 90
            
            if self.use_model and draw_bd_handpose:
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
            
            mode = "[HandPose-X]" if self.use_model else "[Skin Detect]"
            cv2.putText(frame, mode, (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            servo_values = {
                'base': base, 'arm1': arm1, 'arm2': arm2,
                'gripper': grip_angle, 'rotation': rotation, 'gesture': gesture
            }
            
            if self.arm and self.arm.is_connected():
                current_time = time.time()
                if self.is_hand_stable(servo_values['base'], servo_values['arm1']):
                    if current_time - self.last_send_time > MIN_ARM_CONNECT_INTERVAL:
                        try:
                            self.arm.set_all_servos(
                                servo_values['base'], servo_values['arm1'],
                                servo_values['arm2'], servo_values['gripper'],
                                servo_values['rotation']
                            )
                            self.last_send_time = current_time
                        except Exception as e:
                            self.error_logger.add_error("ARM", f"发送指令失败: {e}")
            
            return frame, keypoints, servo_values
        except Exception as e:
            self.error_logger.add_error("PROCESS", f"处理帧失败: {e}")
            return frame, None, None
    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        self.current_frame = frame
        processed_frame, keypoints, servo_values = self.process_frame(frame)
        
        if servo_values:
            self.servo_labels['base'].setText(str(servo_values['base']))
            self.servo_labels['arm1'].setText(str(servo_values['arm1']))
            self.servo_labels['arm2'].setText(str(servo_values['arm2']))
            self.servo_labels['gripper'].setText(str(servo_values['gripper']))
            self.servo_labels['rotation'].setText(str(servo_values['rotation']))
            self.gesture_label.setText(f"手势: {servo_values['gesture']}")
        
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
    
    def start_system(self):
        self.cap = None
        for device_id in [0, 1, 2]:
            try:
                self.cap = cv2.VideoCapture(device_id)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.error_logger.add_error("INFO", f"摄像头打开 device={device_id}")
                        break
                    self.cap.release()
            except Exception as e:
                self.error_logger.add_error("CAMERA", f"打开设备{device_id}失败: {e}")
        
        if self.cap is None or not self.cap.isOpened():
            self.error_logger.add_error("CAMERA", "无法打开摄像头")
            self.refresh_errors()
            return
        
        self.load_arm()
        
        self.video_thread = VideoThread(self.cap)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()
        
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.error_logger.add_error("INFO", "系统启动成功")
        self.refresh_errors()
    
    def stop_system(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.arm:
            self.arm.disconnect()
        
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.error_logger.add_error("INFO", "系统已停止")
        self.refresh_errors()
    
    def clear_errors(self):
        self.error_logger.clear()
        self.error_table.setRowCount(0)
    
    def refresh_errors(self):
        errors = self.error_logger.get_errors()
        self.error_table.setRowCount(len(errors))
        
        for i, err in enumerate(errors):
            self.error_table.setItem(i, 0, QTableWidgetItem(err['timestamp']))
            self.error_table.setItem(i, 1, QTableWidgetItem(err['type']))
            self.error_table.setItem(i, 2, QTableWidgetItem(err['message']))
            self.error_table.setItem(i, 3, QTableWidgetItem(err['details']))
        
        self.error_table.resizeColumnsToContents()
    
    def closeEvent(self, event):
        self.stop_system()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = HandPoseArmGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()