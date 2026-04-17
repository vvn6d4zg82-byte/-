import cv2

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

for i in range(30):
    ret, frame = cap.read()
    print(f"Frame {i}: ret={ret}, shape={frame.shape if ret else None}")
    cv2.imshow("test", frame)
    cv2.waitKey(33)
    
cap.release()
cv2.destroyAllWindows()
print("Done")