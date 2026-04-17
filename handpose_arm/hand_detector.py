import cv2
import numpy as np
import sys

sys.path.insert(0, r'C:\Users\周正\Desktop\33550336\Ayxi\Ayin\handpose_x-main')

def get_skin_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    lower_ycr = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycr = np.array([255, 173, 127], dtype=np.uint8)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask2 = cv2.inRange(ycrcb, lower_ycr, upper_ycr)
    
    mask = cv2.bitwise_and(mask, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 3000:
        return None
    
    return largest

def get_hand_bbox(contour, frame_shape):
    x, y, w, h = cv2.boundingRect(contour)
    
    margin = int(max(w, h) * 0.1)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(frame_shape[1] - x, w + 2 * margin)
    h = min(frame_shape[0] - y, h + 2 * margin)
    
    size = max(w, h)
    
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(frame_shape[1], cx + size // 2)
    y2 = min(frame_shape[0], cy + size // 2)
    
    return (x1, y1, x2, y2)

def detect_hand_bbox(frame):
    mask = get_skin_mask(frame)
    contour = find_largest_contour(mask)
    
    if contour is None:
        return None
    
    bbox = get_hand_bbox(contour, frame.shape)
    return bbox
