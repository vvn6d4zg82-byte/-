import cv2
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode
import time

base_options = base_options_module.BaseOptions(model_asset_path='hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Starting hand detection test...")
print("Will capture 100 frames and report hands found")

found_hand = False
for i in range(100):
    ret, frame = cap.read()
    if not ret:
        continue
    
    if i % 10 == 0:
        print(f"Frame {i}...", end=" ")
    
    mp_image = image_lib.Image(
        image_format=image_lib.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = landmarker.detect(mp_image)
    
    if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
        print(f"HAND FOUND at frame {i}!")
        landmarks = result.hand_landmarks[0]
        hand_x = int(landmarks[9].x * 640)
        hand_y = int(landmarks[9].y * 480)
        print(f"Hand position: ({hand_x}, {hand_y})")
        found_hand = True
        break

cap.release()
landmarker.close()

if found_hand:
    print("\nSUCCESS: MediaPipe hand detection is working!")
else:
    print("\nNo hand detected in 100 frames.")
    print("Possible issues:")
    print("1. Hand not visible in camera")
    print("2. Lighting too dark")
    print("3. Camera not pointing at hand")