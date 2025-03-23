import cv2
from ultralytics import YOLO

model = YOLO("Phát hiện vết nứt trên bề mặt bê tông.pt")

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cam.read()

    result = model.predict(frame)
    new_frame = result[0].numpy()
    boxes = new_frame.boxes.xyxy
    
    for box in boxes:
        x_1 = int(box[0])
        y_1 = int(box[1])
        x_2 = int(box[2])
        y_2 = int(box[3])
        box_img = cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (255,0,0), 2)
        
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


