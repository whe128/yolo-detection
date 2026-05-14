from ultralytics import YOLO
import cv2

# load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# open the camera
cap = cv2.VideoCapture(0)

while True:
    # read each frame image from the camera and detect objects

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # inference stage
    results = model(frame)

    # draw the results on the image
    # result 0 include
    # x1, y1, x2, y2
    # confidence
    # class

    # plot, draw the box automatically
    # boxes gives
    # resykts[0].masks      -> give teh mask of each object
    # results[0].boxes      -> give the box of each object
    #                        results[0].boxes[0].cls[0] -> because it is tensor, only there is one parameter, still need [0]
    # results[0].keypoints  -> give the joint points of each object
    # results[0].orig_img   -> give the original image
    # results[0].names      -> give the name of each class

    # resutls[0].probs      -> give the confidence of each class for each object
    # only used for classification model, that one image belongs to one class
    # detection model, one image can belong to multiple classes
    annotated_frame = results[0].plot()

    # show the image
    cv2.imshow("Detection Results", annotated_frame)

    # refresh duration (1ms), and also check the pressed key
    # if no key is presssed, it will not wait and block the program
    # if waitkey(0), it will wait until a key is pressed, and block the program
    key = cv2.waitKey(1) & 0xFF

    # q for quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

