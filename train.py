from ultralytics import YOLO

# Create YOLOv8 Nano model from scratch (no pre-trained weights)
model = YOLO("yolov8n.yaml", task="detect")

# Train the model
model.train(
    data="data.yaml",   # path to your daStaset config
    epochs=50,          # adjust as needed
    imgsz=640,          # input image size
    batch=16,           # batch sizet
    device="cpu"        # change to "0" for GPU if available
)

print("✅ Training complete. Weights are saved in runs/train/exp/weights/best.pt")
