# Source/Reference : Pysource.com

import cv2
import numpy as np
from ultralytics import YOLO

print(torch.backends.mps.is_available())
cap = cv2.VideoCapture("Video.mp4")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device = "mps")  # device = "mps" to enable gpu(faster), it is cpu by default
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x,y,x2,y2) = bbox
        
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)

       

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)

    if key == 27:
       break

cap.release()
cv2.destroyAllWindows()
