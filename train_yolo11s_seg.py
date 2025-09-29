# train_yolo11s_seg.py
import torch
from ultralytics import YOLO

# ----------------------------
# 🔹 Step 1: Detect GPU/CPU
# ----------------------------
if torch.cuda.is_available():
    device = "cuda"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)} with {vram_gb:.1f} GB VRAM")
else:
    device = "cpu"
    vram_gb = 0
    print("⚠️ No GPU detected. Training will run on CPU (very slow).")

# ----------------------------
# 🔹 Step 2: Set batch size based on VRAM
# ----------------------------
if vram_gb >= 16:
    batch_size = 16
elif vram_gb >= 8:
    batch_size = 8
elif vram_gb >= 4:
    batch_size = 4
else:
    batch_size = 2  # minimal safe batch size

print(f"Using batch size: {batch_size}")

# ----------------------------
# 🔹 Step 3: Load YOLOv11 Small Segmentation Model
# ----------------------------
model = YOLO("yolo11s-seg.pt")

# ----------------------------
# 🔹 Step 4: Train the model
# ----------------------------
results = model.train(
    data="dataset_yolo/data.yaml",  # Path to data.yaml
    epochs=100,                     # Training epochs
    imgsz=640,                      # Image size
    batch=batch_size,               # Auto-adjusted batch size
    device=device,                  # Use GPU/CPU
    project="runs_yolo11s_seg",     # Custom folder
    name="train",                   # Experiment name
)

# ----------------------------
# 🔹 Step 5: Validate the model
# ----------------------------
metrics = model.val()

# ----------------------------
# 🔹 Step 6: Inference on new images
# ----------------------------
predictions = model.predict(
    source="path_to_test_images",   # Replace with your folder of test images
    conf=0.25                       # Confidence threshold
)

print("✅ Training complete! Best weights are saved in:", model.ckpt_path)
