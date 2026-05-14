from ultralytics import YOLO
import cv2

'''
    use the trained model to detect objects in an image
'''

# load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# read the image
image = cv2.imread("test.jpg")

# inference stage
results = model(image)

# draw the results on the image
annotated_frame = results[0].plot()

# show the image
cv2.imshow("Detection Results", annotated_frame)
cv2.waitKey(0)
cv2.destroyALLWindows()
