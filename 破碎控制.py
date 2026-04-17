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

def get_hand_fingers(contour, hull):
    hull = cv2.convexHull(contour, returnPoints=False)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    
    if hull_area == 0:
        return "unknown"
    
    solidity = area / hull_area
    
    try:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return "fist"
        
        defect_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 30 * 256:
                defect_count += 1
        
        if solidity < 0.5:
            return "open"
        elif solidity < 0.7:
            if defect_count >= 3:
                return "open"
            return "half"
        else:
            return "fist"
    except:
        return "fist"

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

bg = None
last_time = time.time()
stable = 0
last_hand = None
last_gesture = None

print("=== CV Hand Tracking ===")
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
        
        if area > 1000:
            x, y, cw, ch = cv2.boundingRect(c)
            cx, cy = x + cw//2, y + ch//2
            
            gesture = get_hand_fingers(c, None)
            
            if last_hand:
                dist = ((cx - last_hand[0])**2 + (cy - last_hand[1])**2)**0.5
                stable = min(stable + 1, 3) if dist < 25 else 0
            
            last_hand = (cx, cy)
            
            hull = cv2.convexHull(c)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            
            new_base = int(15 + 150 * cx / w)
            new_arm = int(165 - 150 * cy / h)
            grip = 180 if gesture == "fist" else 90
            
            cv2.putText(frame, f"B:{new_base} A:{new_arm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"G:{grip} {gesture}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            if stable >= 2 and time.time() - last_time > 0.1:
                move(1, new_base)
                move(3, new_arm)
                move(4, grip)
                last_time = time.time()
                last_gesture = gesture
    else:
        stable = 0
        last_hand = None
    
    if np.random.random() < 0.1:
        bg = gray.copy()
    
    cv2.imshow("Hand", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")