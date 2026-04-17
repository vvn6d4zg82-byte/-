import sys
import os
import cv2
import numpy as np

try:
    import torch
except:
    torch = None

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
ArmController = None

try:
    from models.rexnetv1 import ReXNetV1
    from hand_data_iter.datasets import draw_bd_handpose
except Exception as e:
    print(f"导入handpose模型失败: {e}")

try:
    from hand_detector import detect_hand_bbox
except:
    pass

try:
    from arm_control import ArmController
except:
    pass

from ai_controller.data_collector import DataCollector
from ai_controller.model_trainer import ModelTrainer
from ai_controller.ai_controller import AIController

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QGroupBox, QFrame,
    QSpinBox, QComboBox, QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QStatusBar, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from gesture_utils import (
    recognize_gesture,
    map_position_to_servo,
    get_grip_angle,
    get_palm_center
)

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
        logging.info(f"[{error_type}] {message}")
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


class AIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.mode = "inference"
        self.model_path = MODEL_PATH
        self.model = None
        self.device = None
        self.arm = None
        self.error_logger = ErrorLogger()
        
        self.ai_controller = AIController()
        self.data_collector = DataCollector()
        self.model_trainer = ModelTrainer()
        
        self.use_model = False
        self.running = False
        self.recording = False
        self.cap = None
        self.video_thread = None
        self.current_frame = None
        self.keypoints = None
        self.servo_values = None
        
        self.last_hand_pos = None
        self.stable_count = 0
        self.last_send_time = 0
        
        self.target_servos = {"base": 90, "arm1": 90, "arm2": 90, "gripper": 90, "rotation": 90}
        self.current_gesture = "gesture_1"
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        self.setWindowTitle("AI手势控制机械臂系统")
        self.setGeometry(100, 100, 1400, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout()
        central.setLayout(layout)
        
        left_panel = self.create_video_panel()
        layout.addWidget(left_panel, 3)
        
        right_panel = self.create_control_panel()
        layout.addWidget(right_panel, 2)
    
    def create_video_panel(self):
        group = QGroupBox("视频显示")
        layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        info_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_layout.addWidget(self.fps_label)
        
        self.mode_label = QLabel("模式: 推理")
        self.mode_label.setStyleSheet("font-size: 14px; color: #00AAFF;")
        info_layout.addWidget(self.mode_label)
        
        self.status_label = QLabel("状态: 空闲")
        self.status_label.setStyleSheet("font-size: 14px; color: #FFAA00;")
        info_layout.addWidget(self.status_label)
        
        layout.addLayout(info_layout)
        
        group.setLayout(layout)
        return group
    
    def create_control_panel(self):
        group = QGroupBox("控制面板")
        layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["推理", "数据收集", "训练"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)
        
        btn_style = "QPushButton { padding: 10px; font-size: 14px; margin: 5px; }"
        
        self.start_btn = QPushButton("启动")
        self.start_btn.setStyleSheet(btn_style)
        self.start_btn.clicked.connect(self.start_system)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet(btn_style)
        self.stop_btn.clicked.connect(self.stop_system)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        layout.addWidget(self.create_servo_panel())
        
        layout.addWidget(self.create_collect_panel())
        
        layout.addWidget(self.create_log_panel())
        
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_servo_panel(self):
        group = QGroupBox("舵机状态")
        layout = QVBoxLayout()
        
        self.servo_labels = {}
        for key, label in [("base", "S1 基座"), ("arm1", "S2 大臂"), 
                         ("arm2", "S3 小臂"), ("gripper", "S4 夹爪"), ("rotation", "S5 旋转")]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(70)
            val_lbl = QLabel("90")
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
    
    def create_collect_panel(self):
        group = QGroupBox("数据收集")
        layout = QVBoxLayout()
        
        self.record_btn = QPushButton("开始记录")
        self.record_btn.setStyleSheet("QPushButton { padding: 8px; background: #FF6666; }")
        self.record_btn.clicked.connect(self.toggle_record)
        self.record_btn.setEnabled(False)
        layout.addWidget(self.record_btn)
        
        servo_layout = QHBoxLayout()
        for key, label in [("base", "S1"), ("arm1", "S2"), ("arm2", "S3"), 
                         ("gripper", "S4"), ("rotation", "S5")]:
            btn_layout = QVBoxLayout()
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            btn_layout.addWidget(lbl)
            
            minus_btn = QPushButton("-")
            minus_btn.clicked.connect(lambda checked, k=key: self.adjust_servo(k, -5))
            btn_layout.addWidget(minus_btn)
            
            plus_btn = QPushButton("+")
            plus_btn.clicked.connect(lambda checked, k=key: self.adjust_servo(k, 5))
            btn_layout.addWidget(plus_btn)
            
            servo_layout.addLayout(btn_layout)
        
        layout.addLayout(servo_layout)
        
        group.setLayout(layout)
        return group
    
    def create_log_panel(self):
        group = QGroupBox("日志")
        layout = QVBoxLayout()
        
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["时间", "类型", "消息"])
        self.log_table.setMinimumHeight(150)
        layout.addWidget(self.log_table)
        
        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(clear_btn)
        
        save_btn = QPushButton("保存日志")
        save_btn.clicked.connect(self.save_log)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
        group.setLayout(layout)
        return group
    
    def change_mode(self, mode_text):
        self.mode = mode_text.lower()
        
        if self.mode == "推理":
            self.mode_label.setText("模式: 推理")
        elif self.mode == "数据收集":
            self.mode_label.setText("模式: 数据收集")
        elif self.mode == "训练":
            self.mode_label.setText("模式: 训练")
        
        self.error_logger.add_error("MODE", f"切换到{mode_text}模式")
        self.refresh_log()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path) and ReXNetV1 is not None and torch is not None:
                self.model = ReXNetV1(num_classes=NUM_CLASSES)
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(self.device)
                self.model.eval()
                chkpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(chkpt)
                self.use_model = True
                self.error_logger.add_error("INFO", "HandPose-X模型加载成功")
        except Exception as e:
            self.error_logger.add_error("MODEL", f"加载模型失败: {e}")
            self.use_model = False
        
        self.ai_controller.load_model()
        self.refresh_log()
    
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
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        bbox = detect_hand_bbox(frame) if detect_hand_bbox else None
        
        if bbox is None:
            self.stable_count = 0
            self.last_hand_pos = None
            return frame, None, None
        
        keypoints = self.estimate_keypoints(bbox, w, h)
        
        if self.mode == "推理":
            servo_output = self.ai_controller.predict_with_confidence(keypoints)
            if isinstance(servo_output, tuple):
                servos, confidence = servo_output
            else:
                servos = servo_output
                confidence = "low"
        else:
            try:
                palm_center = get_palm_center(keypoints)
                hand_x = int(palm_center['x'])
                hand_y = int(palm_center['y'])
                gesture = recognize_gesture(keypoints, w, h)
            except:
                hand_x, hand_y = w // 2, h // 2
                gesture = "未知"
            
            base, arm1, arm2, rotation = map_position_to_servo(hand_x, hand_y, w, h)
            grip_angle = get_grip_angle(keypoints, w, h)
            
            servos = {
                'base': base,
                'arm1': arm1,
                'arm2': arm2,
                'gripper': grip_angle,
                'rotation': rotation,
                'gesture': gesture
            }
            confidence = "manual"
        
        if draw_bd_handpose:
            draw_bd_handpose(frame, keypoints, 0, 0)
        
        for i in range(21):
            x = int(keypoints[str(i)]["x"])
            y = int(keypoints[str(i)]["y"])
            cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
        
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        cv2.putText(frame, f"S1:{servos.get('base', 90)} S2:{servos.get('arm1', 90)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S3:{servos.get('arm2', 90)} S5:{servos.get('rotation', 90)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"S4:{servos.get('gripper', 90)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        status = self.mode
        if self.recording:
            status += " [REC]"
        cv2.putText(frame, status, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if self.mode == "数据收集" and self.recording and keypoints:
            real_servos = self.target_servos.copy()
            real_servos['gesture'] = self.current_gesture
            self.data_collector.add_sample(keypoints, real_servos)
        
        return frame, keypoints, servos
    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        self.current_frame = frame
        processed_frame, keypoints, servo_values = self.process_frame(frame)
        
        if servo_values:
            self.servo_labels['base'].setText(str(servo_values.get('base', 90)))
            self.servo_labels['arm1'].setText(str(servo_values.get('arm1', 90)))
            self.servo_labels['arm2'].setText(str(servo_values.get('arm2', 90)))
            self.servo_labels['gripper'].setText(str(servo_values.get('gripper', 90)))
            self.servo_labels['rotation'].setText(str(servo_values.get('rotation', 90)))
            self.gesture_label.setText(f"手势: {servo_values.get('gesture', '---')}")
        
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
            self.refresh_log()
            return
        
        try:
            self.arm = ArmController()
            self.arm.connect()
            self.error_logger.add_error("INFO", "机械臂连接成功")
        except:
            pass
        
        self.video_thread = VideoThread(self.cap)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()
        
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        if self.mode == "数据收集":
            self.record_btn.setEnabled(True)
        
        self.status_label.setText("状态: 运行中")
        self.error_logger.add_error("INFO", "系统启动成功")
        self.refresh_log()
    
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
        self.recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        
        self.status_label.setText("状态: 空闲")
        self.error_logger.add_error("INFO", "系统已停止")
        self.refresh_log()
    
    def toggle_record(self):
        if self.recording:
            self.data_collector.end_demo()
            self.recording = False
            self.record_btn.setText("开始记录")
            self.record_btn.setStyleSheet("QPushButton { padding: 8px; background: #FF6666; }")
        else:
            self.data_collector.start_demo(self.current_gesture)
            self.recording = True
            self.record_btn.setText("停止记录")
            self.record_btn.setStyleSheet("QPushButton { padding: 8px; background: #66FF66; }")
    
    def adjust_servo(self, key, delta):
        current = self.target_servos.get(key, 90)
        new_value = max(0, min(180, current + delta))
        self.target_servos[key] = new_value
        self.servo_labels[key].setText(str(new_value))
    
    def clear_log(self):
        self.error_logger.clear()
        self.log_table.setRowCount(0)
    
    def refresh_log(self):
        errors = self.error_logger.get_errors()
        self.log_table.setRowCount(len(errors))
        
        for i, err in enumerate(errors):
            self.log_table.setItem(i, 0, QTableWidgetItem(err['timestamp']))
            self.log_table.setItem(i, 1, QTableWidgetItem(err['type']))
            self.log_table.setItem(i, 2, QTableWidgetItem(err['message']))
        
        self.log_table.resizeColumnsToContents()
    
    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存日志", "log.txt", "Text (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                for err in self.error_logger.get_errors():
                    f.write(f"{err['timestamp']} [{err['type']}] {err['message']}\n")
            QMessageBox.information(self, "保存", f"日志已保存到 {path}")
    
    def closeEvent(self, event):
        self.stop_system()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = AIGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()