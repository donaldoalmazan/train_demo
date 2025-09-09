# Train Bogie Object Detection & Tracking Demo

This project demonstrates a **train bogie (wheel assembly) detection and counting application** that runs locally on a Raspberry Pi 5B 8GB using a webcam. 

This prototype runs was created using a model train set and is designed to scale up for real-world train monitoring.

## Quick Start

Navigate to the project folder (~/dev/train_demo/) on the Pi, either on terminal or VSCode, and run the following code:

```bash
python bogie_counter_yolo.py
```
Usage Notes:
- Bogie count is shown on-screen and logged in the console.
- Make sure webcam is focused by placing an object behind the train to limit autofocus.
- Press ESC to exit the viewer. 

## How it works

This project uses OpenCV and a custom trained Ultralytics YOLOv8n model. 

The [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [ByteTrack](https://github.com/ifzhang/ByteTrack) tracking libraries. And OpenCV for webcam interfacing & manipulation.

The model was created using RoboFlow to annotate a dataset based on a video of the train, then Ultralytics Hub to train a Yolov8n model, more information on training procedure below.

## Installation Procedure
To install this on a new Raspberry Pi or adapting the code.

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/train_demo.git
   cd train_demo
   
2. **Create environment & install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Download a trained YOLO model**

Use the provided toy demo weights (yolov8n.pt / yolov11n.pt)

Or your custom bogie-trained model (place it in weights/bogie_best.pt)

Run the bogie counter on a video

```bash

Copy code
python bogie_counter_yolo_crossing.py
```


## Training a custom model

1. Upload demo video to [Roboflow](https://roboflow.com/), annotate the sample with a single class 'bogie', and export YOLO dataset as a zip file
2. Import the zip to [Ultralytics Hub](https://hub.ultralytics.com/home), train a YOLOv8n model and download the .pt file once complete
3. Replace MODEL_PATH in the script with your trained weights, .pt file
4. Re-run the counter for improved results


