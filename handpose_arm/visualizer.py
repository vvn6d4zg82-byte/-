import cv2
import numpy as np

KEYPOINT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

COLORS = [
    (0, 215, 255),
    (255, 115, 55),
    (5, 255, 55),
    (25, 15, 255),
    (225, 15, 55)
]

def draw_skeleton(img, keypoints, thickness=2):
    for i, connection in enumerate(KEYPOINT_CONNECTIONS):
        p1 = keypoints[str(connection[0])]
        p2 = keypoints[str(connection[1])]
        
        color = COLORS[i // 4] if i < 20 else COLORS[0]
        
        cv2.line(img, 
                 (int(p1['x']), int(p1['y'])),
                 (int(p2['x']), int(p2['y'])),
                 color, thickness)

def draw_keypoints(img, keypoints, radius=3, color=(255, 50, 60)):
    for i in range(21):
        x = int(keypoints[str(i)]['x'])
        y = int(keypoints[str(i)]['y'])
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.circle(img, (x, y), radius - 2, (255, 150, 180), -1)

def draw_fingertips(img, keypoints, radius=5, color=(0, 255, 255)):
    fingertip_indices = [4, 8, 12, 16, 20]
    
    for idx in fingertip_indices:
        x = int(keypoints[str(idx)]['x'])
        y = int(keypoints[str(idx)]['y'])
        cv2.circle(img, (x, y), radius, color, -1)

def draw_palm_center(img, keypoints, radius=6, color=(0, 255, 0)):
    palm_idx = 9
    x = int(keypoints[str(palm_idx)]['x'])
    y = int(keypoints[str(palm_idx)]['y'])
    cv2.circle(img, (x, y), radius, color, -1)

def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_info_text(img, text, position=(10, 30), font_scale=0.6, 
                   color=(0, 255, 0), thickness=2):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)

def draw_hand_visualization(img, keypoints, bbox=None, show_skeleton=True,
                           show_keypoints=True, show_fingertips=True):
    if bbox is not None:
        draw_bbox(img, bbox)
    
    if show_skeleton:
        draw_skeleton(img, keypoints)
    
    if show_keypoints:
        draw_keypoints(img, keypoints)
    
    if show_fingertips:
        draw_fingertips(img, keypoints)
    
    draw_palm_center(img, keypoints)
    
    return img
