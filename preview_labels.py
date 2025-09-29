# preview_labels.py
import os, cv2, glob
import numpy as np
from random import choice

OUT_IMG_DIR = "dataset_yolo/images/train"
OUT_LBL_DIR = "dataset_yolo/labels/train"
PREVIEW_DIR = "preview"
os.makedirs(PREVIEW_DIR, exist_ok=True)

# include jpg, jpeg, png, JPG, JPEG, PNG
samples = []
for ext in ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]:
    samples.extend(glob.glob(os.path.join(OUT_IMG_DIR, ext)))

if not samples:
    print("❌ No images found in", OUT_IMG_DIR)
    print("Run convert_via_to_yolo.py first and check dataset_yolo structure.")
    exit()

sample = choice(samples)
img = cv2.imread(sample)
h, w = img.shape[:2]

lbl = os.path.join(OUT_LBL_DIR, os.path.splitext(os.path.basename(sample))[0] + ".txt")
if not os.path.exists(lbl):
    print("❌ No label file for", sample)
else:
    with open(lbl, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                pts.append((x, y))
            cv2.polylines(img, [np.array(pts, dtype=int)], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(img, str(cls), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out = os.path.join(PREVIEW_DIR, os.path.basename(sample))
    cv2.imwrite(out, img)
    print("✅ Saved preview to", out)
