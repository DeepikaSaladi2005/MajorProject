import os
import uuid
import joblib
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import io, base64
from fpdf import FPDF

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/results"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load models
model1 = YOLO("weights/best.pt")
model2 = YOLO("weights/trained.pt")
severity_model = YOLO("weights/severity_best.pt")
cost_model = joblib.load("weights/repair_cost_model.pkl")

SEVERITY_CLASSES = {0: "Minor", 1: "Moderate", 2: "Severe"}


# ---------------- YOLO Ensemble Detection ----------------
def ensemble_damage_detect(image_path, conf=0.3, iou=0.5):
    results1 = model1.predict(image_path, conf=conf, iou=iou, save=False)[0]
    results2 = model2.predict(image_path, conf=conf, iou=iou, save=False)[0]

    boxes, scores, classes = [], [], []

    def extract(r):
        if r.boxes is not None:
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy()):
                boxes.append(b)
                scores.append(s)
                classes.append(int(c))

    extract(results1)
    extract(results2)

    if not boxes:
        return None

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    classes = torch.tensor(classes)

    keep_idx = torch.ops.torchvision.nms(boxes, scores, iou)
    return {
        "boxes": boxes[keep_idx].numpy(),
        "scores": scores[keep_idx].numpy(),
        "classes": classes[keep_idx].numpy(),
        "names": results1.names
    }


# ---------------- Severity Graph ----------------
def severity_graph(severity_probs):
    fig, ax = plt.subplots()
    ax.bar(list(severity_probs.keys()), list(severity_probs.values()), color=["green", "orange", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Predicted Severity")
    for i, (cls, prob) in enumerate(severity_probs.items()):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha="center", fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64


# ---------------- PDF Report Generation ----------------
def generate_pdf_report(data, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Insurance Claim Report", ln=True, align="C")
    pdf.ln(5)

    # Policyholder
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Policy Holder Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7,
        f"Name: {data['name']}\n"
        f"Phone: {data['phone']}\n"
        f"Email: {data['email']}\n"
        f"Address: {data['address']}"
    )
    pdf.ln(3)

    # Insurance
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Insurance Details", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Policy Number: {data['policy_number']}", ln=True)
    pdf.cell(0, 7, f"Insurance Provider: {data['insurance_provider']}", ln=True)
    pdf.ln(3)

    # Vehicle
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Vehicle Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Car Brand: {data['car']}", ln=True)
    pdf.cell(0, 7, f"Car Model: {data['model']}", ln=True)
    pdf.cell(0, 7, f"Registration Number: {data['registration_number']}", ln=True)
    pdf.ln(3)

    # Damage
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "4. Damage Assessment", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Damaged Part: {data['body_part']}", ln=True)
    pdf.cell(0, 7, f"Predicted Severity: {data['severity']}", ln=True)
    pdf.cell(0, 7, f"Estimated Repair Cost: Rs {data['cost']}", ln=True)

    pdf.output(output_path)


# ---------------- Process Image ----------------
def process_image(image_path, car, model_name, body_part):
    img = cv2.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Detection
    damage_results = ensemble_damage_detect(image_path, conf=0.3, iou=0.4)
    det_img = img.copy()
    if damage_results:
        for (box, cls_id, score) in zip(damage_results["boxes"], damage_results["classes"], damage_results["scores"]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{damage_results['names'][cls_id]} {score:.2f}"
            cv2.putText(det_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    det_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}_det.jpg")
    cv2.imwrite(det_path, det_img)

    # Severity
    severity_results = severity_model.predict(image_path)[0]
    if hasattr(severity_results, "probs") and severity_results.probs is not None:
        probs = severity_results.probs.data.cpu().numpy()
        pred_id = int(np.argmax(probs))
        severity_pred = SEVERITY_CLASSES.get(pred_id, "Unknown")
        severity_probs = {SEVERITY_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    else:
        severity_pred = "Unknown"
        severity_probs = {c: 0.0 for c in SEVERITY_CLASSES.values()}

    graph_img = severity_graph(severity_probs)

    # Heatmap
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    if damage_results:
        for box in damage_results["boxes"]:
            x1, y1, x2, y2 = map(int, box)
            heatmap[y1:y2, x1:x2] += 1
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    heat_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}_heat.jpg")
    cv2.imwrite(heat_path, overlay)

    # Cost estimation
    img_area = img.shape[0] * img.shape[1]
    total_area = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in damage_results["boxes"]) if damage_results else 0
    area_based_cost = int((total_area / img_area) * 100000) if img_area > 0 else 0
    ml_input = {"car": [car], "model": [model_name], "body part": [body_part]}
    ml_cost = int(cost_model.predict(pd.DataFrame(ml_input))[0])
    final_cost = int((area_based_cost + ml_cost) / 2)

    return os.path.basename(det_path), os.path.basename(heat_path), final_cost, severity_pred, graph_img, body_part


# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file uploaded", 400

        car = request.form.get("brand")
        model_name = request.form.get("model")
        reg_no = request.form.get("registration_number")

        unique_id = str(uuid.uuid4())[:8]
        filename = unique_id + "_" + file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        damage_results = ensemble_damage_detect(file_path, conf=0.3, iou=0.4)
        body_part = damage_results["names"][damage_results["classes"][0]] if damage_results else "Unknown"

        det_file, heat_file, cost, severity, graph_img, body_part = process_image(
            file_path, car, model_name, body_part
        )

        # Pass results forward for PDF generation
        return render_template("index.html",
                               det_image=det_file,
                               heat_image=heat_file,
                               cost=cost,
                               severity=severity,
                               graph_img=graph_img,
                               car=car,
                               model=model_name,
                               reg_no=reg_no,
                               body_part=body_part)
    return render_template("index.html")


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.form.to_dict()
    base_name = str(uuid.uuid4())[:8]
    pdf_filename = f"{base_name}_report.pdf"
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)
    generate_pdf_report(data, pdf_path)
    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)




'''<!DOCTYPE html>
<html>
<head>
    <title>FixMyRide - Damage Detection</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        .container { max-width: 900px; margin: auto; }
        form { text-align: left; background: #f9f9f9; padding: 20px; border-radius: 10px; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input, select, textarea, button { width: 100%; margin-top: 5px; padding: 8px; font-size: 14px; }
        img { max-width: 100%; border-radius: 8px; margin-top: 10px; }
        .result { margin-top: 20px; }
        .comparison { display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; }
        .comparison div { flex: 1 1 45%; }
        h2 { margin-top: 20px; }
        .severity-minor { color: green; font-weight: bold; }
        .severity-moderate { color: orange; font-weight: bold; }
        .severity-severe { color: red; font-weight: bold; }
        .download-btn { margin-top: 15px; padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        .download-btn:hover { background-color: #0056b3; }
    </style>
    <script>
        const brandModels = {
            "HONDA": ["City", "Amaze", "WR-V", "Jazz", "HR-V", "Pilot", "CR-V", "Accord", "Civic"],
            "MARUTI SUZUKI": ["Swift", "Baleno", "Vitara Brezza", "Wagon R", "Ertiga", "Grand Vitara"],
            "TOYOTA": ["Corolla", "Camry", "Fortuner", "Innova", "Yaris"],
            "HYUNDAI": ["i20", "Creta", "Verna", "Venue", "Tucson"],
            "NISSAN": ["Altima", "Rogue", "Sentra", "Pathfinder", "Titan"],
            "SKODA": ["Octavia", "Superb", "Rapid", "Kodiaq", "Karoq"]
        };

        function updateModels() {
            const brandSelect = document.getElementById("brand");
            const modelSelect = document.getElementById("model");
            const selectedBrand = brandSelect.value;
            modelSelect.innerHTML = "";
            if (selectedBrand && brandModels[selectedBrand]) {
                brandModels[selectedBrand].forEach(model => {
                    const option = document.createElement("option");
                    option.value = model;
                    option.text = model;
                    modelSelect.add(option);
                });
            }
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>🚗 FixMyRide - Vehicle Damage Detection</h1>
        
        <!-- Upload form -->
        <form method="post" enctype="multipart/form-data">
            <h2>Upload Vehicle Image</h2>
            <label for="brand">Select Car Brand:</label>
            <select id="brand" name="brand" onchange="updateModels()" required>
                <option value="">--Select Brand--</option>
                <option value="HONDA">HONDA</option>
                <option value="MARUTI SUZUKI">MARUTI SUZUKI</option>
                <option value="TOYOTA">TOYOTA</option>
                <option value="HYUNDAI">HYUNDAI</option>
                <option value="NISSAN">NISSAN</option>
                <option value="SKODA">SKODA</option>
            </select>

            <label for="model">Select Car Model:</label>
            <select id="model" name="model" required>
                <option value="">--Select Model--</option>
            </select>

            <!--<label for="registration_number">Registration Number:</label>
            <input type="text" id="registration_number" name="registration_number" placeholder="Enter car registration number" required>-->

            <label for="file">Upload Vehicle Image:</label>
            <input type="file" name="file" accept="image/*" required>

            <button type="submit">Upload & Analyze</button>
        </form>

        {% if det_image %}
        <div class="result">
            <h2>Detection Result & Damage Analysis</h2>
            <div class="comparison">
                <div>
                    <h3>Original with Detection</h3>
                    <img src="{{ url_for('static', filename='results/' + det_image) }}" alt="Detection">
                </div>
                <div>
                    <h3>Damage Heatmap</h3>
                    <img src="{{ url_for('static', filename='results/' + heat_image) }}" alt="Heatmap">
                </div>
            </div>

            <h2>💰 Estimated Repair Cost: ₹{{ cost }}</h2>
            <h2>
                🔥 Predicted Severity:
                <span class="
                    {% if severity == 'Minor' %}severity-minor
                    {% elif severity == 'Moderate' %}severity-moderate
                    {% elif severity == 'Severe' %}severity-severe
                    {% endif %}">
                    {{ severity }}
                </span>
            </h2>
            <h2>📊 Severity Graph</h2>
            <img src="data:image/png;base64,{{ graph_img }}" alt="Severity Graph" />

            <!-- PDF Form (appears only after analysis) -->
            <h2>Fill in Details to Generate Insurance Claim PDF</h2>
            <form method="post" action="{{ url_for('generate_pdf') }}">
                <label for="name">Full Name:</label>
                <input type="text" name="name" required>

                <label for="address">Address:</label>
                <textarea name="address" rows="3" required></textarea>

                <label for="phone">Phone:</label>
                <input type="text" name="phone" required>

                <label for="email">Email:</label>
                <input type="email" name="email" required>

                <label for="policy_number">Policy Number:</label>
                <input type="text" name="policy_number" required>

                <label for="insurance_provider">Insurance Provider:</label>
                <input type="text" name="insurance_provider" required>

                <!-- Hidden fields with analysis results -->
                <input type="hidden" name="car" value="{{ car }}">
                <input type="hidden" name="model" value="{{ model }}">
                <input type="hidden" name="registration_number" value="{{ reg_no }}">
                <input type="hidden" name="body_part" value="{{ body_part }}">
                <input type="hidden" name="severity" value="{{ severity }}">
                <input type="hidden" name="cost" value="{{ cost }}">

                <button type="submit" class="download-btn">Download Insurance Claim PDF</button>
            </form>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''