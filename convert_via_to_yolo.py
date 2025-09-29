import json
import os
import shutil
from tqdm import tqdm
from glob import glob
from PIL import Image

# Update these paths
TRAIN_JSON = "VehiDE_Dataset/0Train_via_annos.json"
VAL_JSON   = "VehiDE_Dataset/0Val_via_annos.json"
TRAIN_IMG_DIR = "VehiDE_Dataset/image/image"
VAL_IMG_DIR   = "VehiDE_Dataset/validation/validation"

OUT_DIR = "dataset_yolo"
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_LABELS = os.path.join(OUT_DIR, "labels")

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

def convert_json(json_path, img_dir, split, class_map):
    out_img_dir = os.path.join(OUT_IMAGES, split)
    out_lbl_dir = os.path.join(OUT_LABELS, split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    for fname, item in tqdm(data.items(), desc=f"Converting {split}"):
        filename = item["name"]
        regions = item.get("regions", [])

        src_img = os.path.join(img_dir, filename)
        if not os.path.exists(src_img):
            matches = glob(os.path.join(img_dir, "**", filename), recursive=True)
            if matches:
                src_img = matches[0]
            else:
                continue  # skip missing image

        # copy image
        dst_img = os.path.join(out_img_dir, filename)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        # open image to get size
        try:
            w, h = Image.open(src_img).size
        except:
            continue

        lbl_path = os.path.join(out_lbl_dir, filename.rsplit(".", 1)[0] + ".txt")
        with open(lbl_path, "w") as lf:
            for r in regions:
                xs = r.get("all_x", [])
                ys = r.get("all_y", [])
                if not xs or not ys or len(xs) != len(ys):
                    continue

                cls_name = r.get("class", "damage")
                cls_id = class_map.setdefault(cls_name, len(class_map))

                norm_points = []
                for x, y in zip(xs, ys):
                    norm_points.extend([x / w, y / h])

                if len(norm_points) >= 6:
                    lf.write(str(cls_id) + " " + " ".join(f"{p:.6f}" for p in norm_points) + "\n")

def build_class_map(*json_paths):
    classes = {}
    for jp in json_paths:
        data = json.load(open(jp))
        for _, item in data.items():
            for r in item.get("regions", []):
                cname = r.get("class")
                if cname and cname not in classes:
                    classes[cname] = len(classes)
    return classes if classes else {"damage": 0}

if __name__ == "__main__":
    class_map = build_class_map(TRAIN_JSON, VAL_JSON)
    print("Class map:", class_map)

    convert_json(TRAIN_JSON, TRAIN_IMG_DIR, "train", class_map)
    convert_json(VAL_JSON, VAL_IMG_DIR, "val", class_map)

    # Write data.yaml
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(f"train: {os.path.join(OUT_IMAGES, 'train')}\n")
        f.write(f"val: {os.path.join(OUT_IMAGES, 'val')}\n\n")
        f.write("names:\n")
        for cname, cid in sorted(class_map.items(), key=lambda x: x[1]):
            f.write(f"  {cid}: {cname}\n")

    print("✅ Conversion complete! YOLO dataset saved in", OUT_DIR)
