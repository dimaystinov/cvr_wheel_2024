from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(1)
#cap.set(3, 640)
#cap.set(4, 480)

while True:
    ret, img = cap.read()

    if not ret:
        break
    w, h, _ = img.shape
    results = model.predict(img)

    for r in results:

        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls

            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    # img = cv2.resize(img, (2 * h,2 * w ))
    cv2.imshow('YOLO V8 Detection', img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
