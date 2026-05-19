# YOLOv8 Object Detection (Webcam)

A compact YOLOv8-based object detection project for training on a custom dataset (exported from Roboflow) and running real-time inference with a webcam.

This repository contains:
- `train.py` — training script (uses Ultralytics YOLOv8)
- `detect.py` — detection utilities
- `webcam.py` — realtime webcam inference using the trained `best.pt`
- `datasets/` — images and labels (YOLO format)
- `runs/` — training outputs (models and logs)

Project flow (what I did):
1. Collected images and labeled them in Roboflow.
2. Exported the dataset as a YOLOv8 / YOLO format archive and placed files under `datasets/`.
3. Completed `data.yaml` pointing to the `train`/`val` folders and the class names.
4. Ran training to produce `runs/detect/train/weights/best.pt`.
5. Used `best.pt` with `webcam.py` to detect objects live from a camera.

Quick start
-----------
Prerequisites (Python 3.8+ recommended):

Install required packages:

```bash
pip install -U ultralytics opencv-python
```

(Optional GPU) If you want GPU acceleration, install a compatible `ultralytics` wheel and CUDA toolkit according to the Ultralytics docs.

Prepare your dataset
--------------------
Your dataset should follow YOLO folder structure (example used here):

```
datasets/
	images/
		train/
		val/
	labels/
		train/
		val/
```

Example `data.yaml` (place at project root or update paths accordingly):

```yaml
train: datasets/images/train
val: datasets/images/val
nc: 2
names: ['class1', 'class2']
```

- Set `nc` to the number of classes in your dataset.
- Replace the `names` list with your actual class names.

Training
--------
You can train using the provided `train.py` or the Ultralytics CLI. Example commands:

Using the Ultralytics CLI (if `yolo` is on PATH):

```bash
# replace yolov8n.pt with your backbone or config if needed
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

Using the Python API (example):

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=640)
```

After training, your best weights will be saved under `runs/detect/train/weights/best.pt` (this repo expects that path).

Inference with webcam
---------------------
`webcam.py` in this repo is configured to load `runs/detect/train/weights/best.pt`:

```python
model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(1)  # change index to 0 if needed
```

Run realtime detection:

```bash
python webcam.py
```

Notes:
- Camera index: Change `cv2.VideoCapture(1)` to `0` (or another index) depending on your system. If using a phone camera over USB or IP stream, update the capture URL accordingly.
- If frames fail to grab, check camera permissions and that the right index/URL is used.

Tips & troubleshooting
----------------------
- If no objects are detected but validation shows good metrics, verify the `names` order in `data.yaml` matches your label ids.
- If training is slow, enable GPU support and ensure CUDA/cuDNN are installed and the correct PyTorch/Ultralytics build is used.
- To inspect training logs and predictions, check `runs/detect/train/` for `results.png`, `events` and `.pt` files.

File references
---------------
- See `webcam.py` for the inference script.
- See `train.py` for the training entrypoint.

License
-------
Use as you like. Add a license file if needed.

Enjoy detecting! If you want, I can also add a `requirements.txt` or a sample `data.yaml` filled with your actual class names — tell me your class list and I'll add it.
