from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/exp/weights/best.pt")  # path to best weights

# Run inference on a single image or folder of images
results = model.predict(
    source="assets/test.jpg",  # replace with your test image or folder
    conf=0.5,                  # confidence threshold
    save=True,                 # save images with bounding boxes to runs/predict/
    show=True                   # display image during prediction
)

# Optional: print results info
print(results)
