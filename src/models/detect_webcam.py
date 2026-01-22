import cv2
from ultralytics import YOLO

model = YOLO('weights/ecovision_waste_v1/weights/best.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("error")
    exit()

print("Webcam started !")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow('EcoVision - Real Time Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()