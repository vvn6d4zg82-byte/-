import os
import cv2
import numpy as np
import json

print("=== 转换视频数据为训练格式 ===")

input_file = r"C:\Users\周正\Videos\Screen Recordings\Screen Recording 2026-04-16 125630_keyframes.json"

if not os.path.exists(input_file):
    print(f"错误: 文件不存在 {input_file}")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"加载帧数: {len(data['frames'])}")

frames = data["frames"]

def keypoints_to_array(keypoints):
    arr = []
    for i in range(21):
        kp = keypoints.get(str(i), {"x": 0, "y": 0})
        arr.append(kp["x"])
        arr.append(kp["y"])
    return np.array(arr, dtype=np.float32)

X = []
timestamps = []

target_columns = [0, 60, 120, 180, 240, 300]
servo_values = [
    [90, 90, 90, 90, 90],
    [90, 90, 90, 30, 90],
    [90, 90, 90, 150, 90],
    [90, 90, 150, 90, 90],
    [90, 90, 30, 90, 90],
    [90, 150, 90, 90, 90],
]

print("按时间分段...")
print(f"  视频时长: {frames[-1]['time']:.1f}秒")

start_time = 0
segments = [
    (0, 30, "close_gripper"),
    (30, 60, "open_gripper"),
    (60, 90, "move_up"),
    (90, 120, "move_down"),
    (120, 150, "base_left"),
    (150, 180, "base_right"),
    (180, 210, "arm_up"),
    (210, 240, "arm_down"),
]

demo = {
    "name": "video_demo",
    "timestamp": data.get("timestamp", ""),
    "samples": []
}

for frame in frames:
    t = frame["time"]
    kp_array = keypoints_to_array(frame["keypoints"])
    
    idx = int(t) // 30
    if idx >= len(servo_values):
        idx = len(servo_values) - 1
    
    sample = {
        "keypoints": kp_array.tolist(),
        "servos": servo_values[idx]
    }
    demo["samples"].append(sample)
    X.append(kp_array)

X = np.array(X, dtype=np.float32)
y = np.array([servo_values[int(f["time"]) // 30 if int(f["time"]) // 30 < len(servo_values) else len(servo_values) - 1] for f in frames], dtype=np.float32)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

os.makedirs("data", exist_ok=True)
output_file = "data/video_demonstrations.json"

demos = [demo]
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(demos, f, ensure_ascii=False, indent=2)

print(f"\n保存到: {output_file}")
print(f"总样本: {len(demo['samples'])}")

print("\n样本舵机值统计:")
servo_names = ["base", "arm1", "arm2", "gripper", "rotation"]
for i, name in enumerate(servo_names):
    unique = sorted(set(y[:, i].astype(int)))
    print(f"  {name}: {unique}")