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

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    use_mediapipe = True
    print("MediaPipe loaded")
except:
    use_mediapipe = False
    print("MediaPipe not available")

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
last_time = time.time()
stable = 0
last_hand = None
last_gesture = None

def get_finger_count(hand_landmarks):
    fingers = []
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
        fingers.append(1)
    return sum(fingers)

def get_hand_gesture(hand_landmarks):
    fingers = get_finger_count(hand_landmarks)
    if fingers == 0:
        return "fist"
    elif fingers == 5:
        return "open"
    elif fingers == 1:
        return "point"
    elif fingers == 2:
        return "peace"
    return f"{fingers}fingers"

print("=== MediaPipe Gesture Control ===")
print("S1:Base X  S2:Arm Y  S3:Arm height")
print("S4:Grip  S5:Rot X")
print("Q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    if use_mediapipe:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            
            gesture = get_hand_gesture(hand_landmarks)
            fingers = get_finger_count(hand_landmarks)
            
            if last_hand:
                dist = ((cx - last_hand[0])**2 + (cy - last_hand[1])**2)**0.5
                if dist < 15:
                    stable = min(stable + 1, 3)
                else:
                    stable = 0
            
            last_hand = (cx, cy)
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            
            new_base = int(15 + 150 * cx / w)
            new_arm = int(15 + 150 * cy / h)
            new_arm2 = int(165 - 150 * cy / h)
            new_rot = int(180 - 180 * cx / w)
            grip_angle = 180 if gesture == "fist" else 90
            
            if stable >= 2 and time.time() - last_time > 0.08:
                move(1, new_base)
                move(2, new_arm)
                move(3, new_arm2)
                move(5, new_rot)
                move(4, grip_angle)
                last_time = time.time()
                last_gesture = gesture
            
            cv2.putText(frame, f"S1:{new_base} S2:{new_arm}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S3:{new_arm2} S5:{new_rot}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"S4:{gesture} ({fingers})", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            stable = 0
            last_hand = None
            last_gesture = None
    else:
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
                
                if last_hand:
                    dist = ((cx - last_hand[0])**2 + (cy - last_hand[1])**2)**0.5
                    if dist < 20:
                        stable = min(stable + 1, 3)
                    else:
                        stable = 0
                
                last_hand = (cx, cy)
                
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
            last_hand = None
        
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