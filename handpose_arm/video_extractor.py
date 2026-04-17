import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime

print("=== 视频手势数据提取 ===")
print(f"视频路径: {sys.argv[1] if len(sys.argv) > 1 else '未指定'}")
print("用法: python video_extractor.py <视频文件路径>")

sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')

try:
    from hand_detector import detect_hand_bbox
except:
    detect_hand_bbox = None
    print("警告: hand_detector 导入失败")


def estimate_keypoints(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1)
    
    keypoints = {"0": {"x": cx, "y": cy}}
    
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0, 45, 90, 135, 180, 225, 270, 315, 0]
    radii = [0, 0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.98, 0.1, 0.25, 0.4, 0.55, 0.7, 0.82, 0.9, 0.95, 0.0]
    
    for i in range(21):
        angle_rad = angles[i] * np.pi / 180
        radius = size * radii[i]
        x = cx + radius * np.cos(angle_rad)
        y = cy + radius * np.sin(angle_rad)
        keypoints[str(i)] = {"x": x, "y": y}
    
    return keypoints


def process_video(video_path, output_path=None, skip_frames=2):
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return None
    
    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + "_keyframes.json"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")
    
    data = {
        "video": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    frame_idx = 0
    processed = 0
    detected = 0
    
    print("处理中...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % (skip_frames + 1) == 0:
            h, w = frame.shape[:2]
            
            bbox = None
            if detect_hand_bbox:
                bbox = detect_hand_bbox(frame)
            
            if bbox:
                keypoints = estimate_keypoints(bbox, w, h)
                
                frame_data = {
                    "frame": frame_idx,
                    "time": frame_idx / fps,
                    "bbox": list(bbox),
                    "keypoints": keypoints
                }
                data["frames"].append(frame_data)
                detected += 1
                
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                cv2.circle(frame, (cx, cy), 30, (0, 255, 0), 2)
            
            processed += 1
            
            if frame_idx % 30 == 0:
                print(f"  帧 {frame_idx}/{frame_count}, 检测到 {detected} 个手势")
        
        frame_idx += 1
    
    cap.release()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成!")
    print(f"  总帧数: {processed}")
    print(f"  检测到手势: {detected}")
    print(f"  保存到: {output_path}")
    
    return data


if __name__ == "__main__":
    import sys
    
    video_path = r"C:\Users\周正\Videos\Screen Recordings\Screen Recording 2026-04-16 125630.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    output = None
    if len(sys.argv) > 2:
        output = sys.argv[2]
    
    result = process_video(video_path, output)
    
    if result and len(result["frames"]) > 0:
        print("\n提取的前5个关键点:")
        for i, frame in enumerate(result["frames"][:5]):
            kp = frame["keypoints"]
            print(f"  帧{frame['frame']} [{frame['time']:.2f}s]:")
            print(f"    手掌中心: ({kp['0']['x']:.0f}, {kp['0']['y']:.0f})")