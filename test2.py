import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision.core import image as image_lib
from mediapipe.tasks.python.vision.core import vision_task_running_mode

base_options = base_options_module.BaseOptions(model_asset_path='hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Testing with very low threshold...")
print("Show your hand to the camera!")

found = False
for i in range(200):
    ret, frame = cap.read()
    if not ret:
        continue
    
    if i % 20 == 0:
        print(f"Frame {i}")
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = image_lib.Image(
        image_format=image_lib.ImageFormat.SRGB,
        data=rgb
    )
    result = landmarker.detect(mp_image)
    
    if result and result.hand_landmarks:
        print(f"HAND DETECTED at frame {i}!")
        landmarks = result.hand_landmarks[0]
        for j, lm in enumerate(landmarks):
            print(f"  Landmark {j}: ({lm.x:.3f}, {lm.y:.3f})")
        found = True
        break

cap.release()
landmarker.close()

if not found:
    print("\nStill no detection. Trying save image test...")