import os
import uuid
import joblib
from flask import Flask, render_template, request, send_file, redirect, url_for
from flask_babel import Babel, _  # For multilingual support
from ultralytics import YOLO
import cv2
import base64

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from fpdf import FPDF

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/results"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- Babel Config ----------------
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'hi', 'te']

# Locale selector reads form first (POST), then query params (GET)
babel = Babel(app, locale_selector=lambda: request.form.get('lang') or request.args.get('lang') or 'en')

# ---------------- Load Models ----------------
model1 = YOLO("weights/best.pt")
model2 = YOLO("weights/trained.pt")
severity_model = YOLO("weights/severity_best.pt")
cost_model = joblib.load("weights/repair_cost_model.pkl")

SEVERITY_CLASSES = {0: _("Minor"), 1: _("Moderate"), 2: _("Severe")}

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
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import io, base64

    # Set a font that supports Hindi/Telegu/Unicode
    # Make sure this font is installed on your system
    rcParams['font.family'] = 'Noto Sans'  

    fig, ax = plt.subplots(figsize=(6,4))

    # Bar plot
    ax.bar(list(severity_probs.keys()), list(severity_probs.values()), color=["green", "orange", "red"])

    # Labels
    ax.set_ylabel(_("Probability"), fontsize=12)
    ax.set_title(_("Predicted Severity"), fontsize=14)

    # X-axis labels: rotate to avoid overlap
    ax.set_xticks(range(len(severity_probs)))
    ax.set_xticklabels(list(severity_probs.keys()), rotation=30, fontsize=10)

    # Y-axis limits
    ax.set_ylim(0, 1)

    # Add values on top of bars
    for i, (cls, prob) in enumerate(severity_probs.items()):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha="center", fontsize=10)

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()  # Adjust layout to avoid clipping labels
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
    pdf.cell(0, 10, _("Insurance Claim Report"), ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _("1. Policy Holder Information"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7,
        f"{_('Name')}: {data['name']}\n"
        f"{_('Phone')}: {data['phone']}\n"
        f"{_('Email')}: {data['email']}\n"
        f"{_('Address')}: {data['address']}"
    )
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _("2. Insurance Details"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"{_('Policy Number')}: {data['policy_number']}", ln=True)
    pdf.cell(0, 7, f"{_('Insurance Provider')}: {data['insurance_provider']}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _("3. Vehicle Information"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"{_('Car Brand')}: {data['car']}", ln=True)
    pdf.cell(0, 7, f"{_('Car Model')}: {data['model']}", ln=True)
    pdf.cell(0, 7, f"{_('Registration Number')}: {data['registration_number']}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _("4. Damage Assessment"), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"{_('Damaged Part')}: {data['body_part']}", ln=True)
    pdf.cell(0, 7, f"{_('Predicted Severity')}: {data['severity']}", ln=True)
    pdf.cell(0, 7, f"{_('Estimated Repair Cost')}: Rs {data['cost']}", ln=True)

    pdf.output(output_path)

# ---------------- Image Processing ----------------
def process_image(image_path, car, model_name, body_part):
    img = cv2.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    damage_results = ensemble_damage_detect(image_path, conf=0.3, iou=0.4)
    det_img = img.copy()
    if damage_results:
        for (box, cls_id, score) in zip(damage_results["boxes"], damage_results["classes"], damage_results["scores"]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{damage_results['names'][cls_id]} {score:.2f}"
            cv2.putText(det_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    det_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}_det.jpg")
    cv2.imwrite(det_path, det_img)

    # ---------------- Severity ----------------
    severity_results = severity_model.predict(image_path)[0]
    if hasattr(severity_results, "probs") and severity_results.probs is not None:
        probs = severity_results.probs.data.cpu().numpy()
        pred_id = int(np.argmax(probs))
        severity_pred = SEVERITY_CLASSES.get(pred_id, _("Unknown"))
        severity_probs = {SEVERITY_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    else:
        severity_pred = _("Unknown")
        severity_probs = {c: 0.0 for c in SEVERITY_CLASSES.values()}

    # ---------------- Severity Graph ----------------
    graph_img = severity_graph(severity_probs)

    # ---------------- Heatmap ----------------
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

    # ---------------- Cost Estimation ----------------
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
            return _("No file uploaded"), 400

        car = request.form.get("brand")
        model_name = request.form.get("model")
        reg_no = request.form.get("registration_number", "")
        lang = request.form.get("lang", "en")

        unique_id = str(uuid.uuid4())[:8]
        filename = unique_id + "_" + file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        damage_results = ensemble_damage_detect(file_path, conf=0.3, iou=0.4)
        body_part = damage_results["names"][damage_results["classes"][0]] if damage_results else _("Unknown")

        det_file, heat_file, cost, severity, graph_img, body_part = process_image(
            file_path, car, model_name, body_part
        )

        # Redirect to GET with lang param to persist language
        return redirect(url_for('index', 
                                det_image=det_file,
                                heat_image=heat_file,
                                cost=cost,
                                severity=severity,
                                graph_img=graph_img,
                                car=car,
                                model=model_name,
                                reg_no=reg_no,
                                body_part=body_part,
                                lang=lang))
    
    # GET request
    return render_template("index.html",
                           det_image=request.args.get("det_image"),
                           heat_image=request.args.get("heat_image"),
                           cost=request.args.get("cost"),
                           severity=request.args.get("severity"),
                           graph_img=request.args.get("graph_img"),
                           car=request.args.get("car"),
                           model=request.args.get("model"),
                           reg_no=request.args.get("reg_no"),
                           body_part=request.args.get("body_part"))

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
