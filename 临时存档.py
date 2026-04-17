import cv2
import numpy as np
import serial
import time
import sys

try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    time.sleep(2)
except:
    ser = None

def move(s, a):
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass

def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower2 = np.array([0, 133, 77], dtype=np.uint8)
    upper2 = np.array([255, 173, 127], dtype=np.uint8)
    mask2 = cv2.inRange(ycrcb, lower2, upper2)
    
    return cv2.bitwise_and(mask1, mask2)

def get_hand_points(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    
    points = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 25 * 256:
                points.append(tuple(contour[f][0]))
    
    return points

def analyze_hand(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return "none", 0
    
    solidity = area / hull_area
    
    defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
    fingers = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            if defects[i, 0][3] > 25 * 256:
                fingers += 1
    
    # 估算手指数量
    if solidity < 0.45:
        gesture = "open"
    elif solidity < 0.6:
        gesture = f"{fingers}fingers" if fingers > 0 else "half"
    elif solidity < 0.75:
        gesture = "fist"
    else:
        gesture = "fist"
    
    return gesture, fingers

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

bg = None
last_time = time.time()
stable = 0
last_hand = None

print("=== Hand Skeleton Tracking ===")
print("Q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    skin = detect_skin(frame)
    skin = cv2.GaussianBlur(skin, (9, 9), 0)
    
    if bg is None:
        bg = gray.copy()
        continue
    
    diff = cv2.absdiff(gray, bg)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, 3)
    combined = cv2.bitwise_and(thresh, skin)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        if area > 800:
            x, y, cw, ch = cv2.boundingRect(c)
            cx, cy = x + cw//2, y + ch//2
            
            gesture, fingers = analyze_hand(c)
            
            if last_hand:
                dist = ((cx - last_hand[0])**2 + (cy - last_hand[1])**2)**0.5
                stable = min(stable + 1, 3) if dist < 20 else 0
            last_hand = (cx, cy)
            
            hull = cv2.convexHull(c)
            hull_pts = cv2.convexHull(c, returnPoints=True)
            
            # 绘制手部骨架
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [hull_pts], -1, (255, 0, 255), 2)
            
            # 缺陷点（指尖估算）
            defect_pts = get_hand_points(c)
            for pt in defect_pts:
                cv2.circle(frame, pt, 6, (0, 255, 255), -1)
            
            # 绘制指骨线条
            for i in range(len(hull_pts) - 1):
                cv2.line(frame, tuple(hull_pts[i][0]), tuple(hull_pts[i+1][0]), (255, 255, 0), 2)
            
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)
            
            # 绘制手掌中心骨架
            palm_pts = [5, 9, 13, 17]
            for pt_idx in palm_pts:
                if pt_idx < len(c):
                    pt = tuple(c[pt_idx][0])
                    cv2.line(frame, (cx, cy), pt, (255, 100, 100), 1)
            
            new_base = int(15 + 150 * cx / w)
            new_arm = int(15 + 150 * cy / h)
            new_arm2 = int(165 - 150 * cy / h)
            new_rot = int(180 - 180 * cx / w)
            grip = 180 if gesture == "fist" else 90
            
            cv2.putText(frame, f"S1:{new_base} S2:{new_arm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S3:{new_arm2} S5:{new_rot}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S4:{grip} {gesture}({fingers})", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if stable >= 2 and time.time() - last_time > 0.08:
                move(1, new_base)
                move(2, new_arm)
                move(3, new_arm2)
                move(4, grip)
                move(5, new_rot)
                last_time = time.time()
    else:
        stable = 0
        last_hand = None
    
    if np.random.random() < 0.1:
        bg = gray.copy()
    
    cv2.imshow("Hand Skeleton", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")