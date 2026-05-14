from ultralytics import YOLO

def main():
    # load yolo model
    # this is the detect model
    model = YOLO("yolov8n.pt")

    # this is classification model
    # model = YOLO("yolov8n-cls.pt")

    # models
    '''
        yolov8n -> nano, smallest and fastest, least accurate
        yolov8s -> small and fast, less accurate
        yolov8m -> medium, balance between speed and accuracy
        yolov8l -> large, slow but accurate
        yolov8x -> extra large, slowest but most accurate
    '''

    # start training
    model.train(
        data = "data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 8,
        device = 0
    )

if __name__ == "__main__":
    main()
