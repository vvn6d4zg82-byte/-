import os
import cv2
import json
import numpy as np
from datetime import datetime
import time

class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "demonstrations.json")
        os.makedirs(data_dir, exist_ok=True)
        
        self.demonstrations = []
        self.current_demo = None
        self.collecting = False
        self.target_servos = {"base": 90, "arm1": 90, "arm2": 90, "gripper": 90, "rotation": 90}
        
    def start_demo(self, gesture_name="demo"):
        self.current_demo = {
            "name": gesture_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "samples": []
        }
        self.collecting = True
        print(f"[数据收集] 开始采集: {gesture_name}")
        
    def add_sample(self, keypoints, servo_targets):
        if not self.collecting or self.current_demo is None:
            return False
        
        kp_array = self.keypoints_to_array(keypoints)
        
        sample = {
            "keypoints": kp_array.tolist(),
            "servos": [
                servo_targets.get("base", 90),
                servo_targets.get("arm1", 90),
                servo_targets.get("arm2", 90),
                servo_targets.get("gripper", 90),
                servo_targets.get("rotation", 90)
            ]
        }
        
        self.current_demo["samples"].append(sample)
        return True
    
    def keypoints_to_array(self, keypoints):
        arr = []
        for i in range(21):
            kp = keypoints.get(str(i), {"x": 0, "y": 0})
            arr.append(kp["x"])
            arr.append(kp["y"])
        return np.array(arr, dtype=np.float32)
    
    def end_demo(self):
        if self.current_demo is None:
            return
        
        self.collecting = False
        demo = self.current_demo
        
        if len(demo["samples"]) > 0:
            self.demonstrations.append(demo)
            self.save_data()
            print(f"[数据收集] 保存成功: {demo['name']}, {len(demo['samples'])} 个样本")
        else:
            print(f"[数据收集] 放弃: 无样本")
        
        self.current_demo = None
    
    def save_data(self):
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.demonstrations, f, ensure_ascii=False, indent=2)
        print(f"[数据保存] 已保存到 {self.data_file}")
    
    def load_data(self):
        if not os.path.exists(self.data_file):
            return []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.demonstrations = json.load(f)
        print(f"[数据加载] {len(self.demonstrations)} 个演示")
        return self.demonstrations
    
    def get_training_data(self):
        X = []
        y = []
        
        for demo in self.demonstrations:
            for sample in demo["samples"]:
                X.append(sample["keypoints"])
                y.append(sample["servos"])
        
        if len(X) == 0:
            return None, None
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def set_target_servos(self, base=None, arm1=None, arm2=None, gripper=None, rotation=None):
        if base is not None:
            self.target_servos["base"] = base
        if arm1 is not None:
            self.target_servos["arm1"] = arm1
        if arm2 is not None:
            self.target_servos["arm2"] = arm2
        if gripper is not None:
            self.target_servos["gripper"] = gripper
        if rotation is not None:
            self.target_servos["rotation"] = rotation
    
    def get_target_servos(self):
        return self.target_servos.copy()
    
    def clear_data(self):
        self.demonstrations = []
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        print("[数据] 已清空")


class ManualDataCollector:
    """Manual data collection with keyboard controls"""
    
    def __init__(self, collector=None):
        self.collector = collector or DataCollector()
        self.recording = False
        self.current_gesture = "gesture_1"
        
    def run(self):
        print("\n=== 手动数据收集模式 ===")
        print("W/S: 基座 S1 -/+")
        print("A/D: 大臂 S2 -/+")
        print("Q/E: 小臂 S3 -/+")
        print("Z/C: 夹爪 S4 -/+")
        print("R/F: 旋转 S5 -/+")
        print("SPACE: 开始/停止记录当前手势")
        print("N: 新手势名称")
        print("P: 保存并显示数据统计")
        print("Q: 退出")
        print("")
        
        import sys
        sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')
        
        try:
            from hand_detector import detect_hand_bbox
        except:
            detect_hand_bbox = None
        
        self.collector.load_data()
        
        cap = None
        for device_id in [0, 1]:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                break
        
        if cap is None or not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("摄像头已打开")
        
        step = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            if detect_hand_bbox:
                bbox = detect_hand_bbox(frame)
            else:
                bbox = None
            
            if bbox:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                
                keypoints = {}
                keypoints["0"] = {"x": cx, "y": cy}
                for i in range(1, 21):
                    angle = (i * 30) % 360
                    radius = (i % 5 + 1) * 20
                    rad = angle * 3.14159 / 180
                    keypoints[str(i)] = {
                        "x": cx + radius * np.cos(rad),
                        "y": cy + radius * np.sin(rad)
                    }
                
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                
                for i in range(21):
                    x = int(keypoints[str(i)]["x"])
                    y = int(keypoints[str(i)]["y"])
                    cv2.circle(frame, (x, y), 3, (255, 50, 60), -1)
            else:
                keypoints = None
            
            servos = self.collector.get_target_servos()
            
            cv2.putText(frame, f"S1:{servos['base']} S2:{servos['arm1']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S3:{servos['arm2']} S4:{servos['gripper']}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S5:{servos['rotation']}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            status = "REC" if self.recording else "IDLE"
            status += f" | {self.current_gesture}"
            cv2.putText(frame, status, (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.recording else (255, 255, 0), 2)
            
            cv2.imshow("数据收集", frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('w'):
                self.collector.set_target_servos(base=servos['base'] + step)
            elif key == ord('s'):
                self.collector.set_target_servos(base=servos['base'] - step)
            elif key == ord('a'):
                self.collector.set_target_servos(arm1=servos['arm1'] + step)
            elif key == ord('d'):
                self.collector.set_target_servos(arm1=servos['arm1'] - step)
            elif key == ord('q'):
                self.collector.set_target_servos(arm2=servos['arm2'] + step)
            elif key == ord('e'):
                self.collector.set_target_servos(arm2=servos['arm2'] - step)
            elif key == ord('z'):
                self.collector.set_target_servos(gripper=servos['gripper'] + step)
            elif key == ord('c'):
                self.collector.set_target_servos(gripper=servos['gripper'] - step)
            elif key == ord('r'):
                self.collector.set_target_servos(rotation=servos['rotation'] + step)
            elif key == ord('f'):
                self.collector.set_target_servos(rotation=servos['rotation'] - step)
            elif key == ord(' '):
                if self.recording:
                    self.collector.end_demo()
                    self.recording = False
                else:
                    self.collector.start_demo(self.current_gesture)
                    self.recording = True
            elif key == ord('n'):
                name = input("手势名称: ")
                if name:
                    self.current_gesture = name
            elif key == ord('p'):
                demos = self.collector.load_data()
                total = sum(len(d["samples"]) for d in demos)
                print(f"\n=== 数据统计 ===")
                print(f"演示数: {len(demos)}")
                print(f"总样本数: {total}")
                for d in demos:
                    print(f"  {d['name']}: {len(d['samples'])} 样本")
            elif key == ord('q'):
                break
            
            if self.recording and keypoints:
                self.collector.add_sample(keypoints, servos)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.recording:
            self.collector.end_demo()
        
        X, y = self.collector.get_training_data()
        if X is not None:
            print(f"\n=== 最终数据 ===")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")


if __name__ == "__main__":
    collector = DataCollector()
    manual = ManualDataCollector(collector)
    manual.run()