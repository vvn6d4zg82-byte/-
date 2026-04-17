import numpy as np

KEYPOINT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 6, 10, 14, 18]
PALM_CENTER = 9

def calculate_angle(p1, p2):
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle

def calculate_distance(p1, p2):
    return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def get_palm_center(keypoints):
    return keypoints[str(PALM_CENTER)]

def get_fingertips(keypoints):
    return [keypoints[str(i)] for i in FINGER_TIPS]

def get_finger_states(keypoints, frame_width, frame_height):
    states = {}
    
    thumb_tip = keypoints['4']
    thumb_ip = keypoints['3']
    wrist = keypoints['0']
    
    thumb_dist = calculate_distance(thumb_tip, keypoints['1'])
    thumb_angle = abs(calculate_angle(thumb_ip, thumb_tip) - calculate_angle(wrist, thumb_ip))
    
    for i, tip_idx in enumerate(FINGER_TIPS):
        tip = keypoints[str(tip_idx)]
        mcp = keypoints[str(FINGER_MCP[i])]
        
        dist = calculate_distance(tip, get_palm_center(keypoints))
        normalized_dist = dist / frame_width
        
        states[i] = normalized_dist < 0.15
    
    return states

def recognize_gesture(keypoints, frame_width, frame_height):
    palm_center = get_palm_center(keypoints)
    fingertips = get_fingertips(keypoints)
    
    distances = []
    for ft in fingertips:
        d = calculate_distance(ft, palm_center)
        distances.append(d)
    
    avg_dist = sum(distances) / len(distances)
    normalized_avg = avg_dist / frame_width
    
    if normalized_avg < 0.12:
        return "fist"
    
    extended_count = 0
    for ft in fingertips:
        d = calculate_distance(ft, palm_center)
        if d > frame_width * 0.2:
            extended_count += 1
    
    if extended_count == 5:
        return "five"
    
    if extended_count == 1:
        index_tip = keypoints['8']
        if index_tip['y'] < palm_center['y']:
            return "one"
    
    if extended_count == 2:
        index_tip = keypoints['8']
        middle_tip = keypoints['12']
        if index_tip['y'] < palm_center['y'] and middle_tip['y'] < palm_center['y']:
            return "two"
    
    thumb_tip = keypoints['4']
    index_tip = keypoints['8']
    distance_thumb_index = calculate_distance(thumb_tip, index_tip)
    
    if distance_thumb_index < frame_width * 0.08:
        return "pinch"
    
    return "unknown"

def map_position_to_servo(hand_x, hand_y, frame_width, frame_height):
    x_ratio = hand_x / frame_width
    y_ratio = hand_y / frame_height
    
    base = int(15 + 150 * x_ratio)
    arm1 = int(15 + 150 * y_ratio)
    arm2 = int(165 - 150 * y_ratio)
    rotation = int(180 - 180 * x_ratio)
    
    return base, arm1, arm2, rotation

def get_grip_angle(keypoints, frame_width, frame_height):
    palm_center = get_palm_center(keypoints)
    fingertips = get_fingertips(keypoints)
    
    distances = []
    for ft in fingertips:
        d = calculate_distance(ft, palm_center)
        distances.append(d)
    
    avg_dist = sum(distances) / len(distances)
    normalized = avg_dist / frame_width
    
    if normalized < 0.12:
        return 180
    elif normalized < 0.18:
        return 135
    else:
        return 90
