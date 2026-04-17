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

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

bg = None
angles = [90, 90, 90, 90, 90]
last_time = time.time()
stable = 0

print("=== Gesture Control ===")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    skin = detect_skin(frame)
    skin = cv2.GaussianBlur(skin, (15, 15), 0)
    
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
            
            stable += 1
            
            cv2.rectangle(frame, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            
            new_base = int(15 + 150 * cx / w)
            new_arm = int(165 - 150 * cy / h)
            
            if stable >= 2 and time.time() - last_time > 0.08:
                move(1, new_base)
                move(3, new_arm)
                last_time = time.time()
            
            cv2.putText(frame, f"B:{new_base} A:{new_arm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        stable = 0
    
    if np.random.random() < 0.1:
        bg = gray.copy()
    
    cv2.imshow("Gesture", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("Done!")