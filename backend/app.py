# app.py
import os
import uuid
import tempfile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import gradio as gr

# ----------------------------
# 1) Model definition (must match training)
# ----------------------------
class FruitFreshModel(nn.Module):
    def __init__(self, num_fruits=9):
        super().__init__()
        self.alpha = 0.7
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze earlier layers (same as training)
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fruit_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_fruits)
        )

        self.fresh_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        self.shelf_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.base(x)
        fruit_pred = self.fruit_head(features)
        fresh_pred = self.fresh_head(features)
        shelf_pred = self.shelf_head(features)
        return fruit_pred, fresh_pred, shelf_pred

# ----------------------------
# 2) Device + Load weights
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "fruit_freshness_weights_final.pth"  # you specified this earlier

model = FruitFreshModel(num_fruits=9)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Place weights in repo.")

# load checkpoint gracefully (handles both raw state_dict and dict with 'model_state_dict')
ckpt = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.to(DEVICE)
model.eval()

# ----------------------------
# 3) Transforms
# ----------------------------
fruit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# 4) Face detector (Haarcascade)
# ----------------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_human_face(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    faces = face_detector.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=4)
    return len(faces) > 0

# ----------------------------
# 5) HSV-based shelf-life estimation (version 2 you use)
# ----------------------------
def estimate_shelf_life_hsv(image_path, fruit_name):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:,:,0].mean()

    if fruit_name == "Banana":
        return float(max(0, round((120 - h_mean) / 10, 1)))
    elif fruit_name == "Apple":
        return float(max(0, round((150 - h_mean) / 12, 1)))
    elif fruit_name == "Tomato":
        return float(max(0, round((180 - h_mean) / 15, 1)))
    elif fruit_name == "Oranges":
        return float(max(0, round((200 - h_mean) / 16, 1)))

    return 0.0

HSV_SUPPORTED = ["Apple", "Banana", "Tomato", "Oranges"]

def hybrid_shelf_life(predicted_fruit, predicted_freshness, raw_shelf_output, image_path):
    if predicted_freshness == "Rotten":
        return 0.0, "Rotten → Shelf Life = 0 days"

    if predicted_fruit in HSV_SUPPORTED:
        hsv_days = estimate_shelf_life_hsv(image_path, predicted_fruit)
        return float(hsv_days), "HSV Color-Based Estimation"

    # fallback to regression output (clamped)
    reg_days = max(0.0, float(round(raw_shelf_output, 1)))
    return reg_days, "Regression Output"

# ----------------------------
# 6) Labels (must match training order)
# ----------------------------
fruit_labels = [
    'Apple', 'Banana', 'Bittergourd', 'Capsicum',
    'Cucumber', 'Okra', 'Oranges', 'Potato', 'Tomato'
]
fresh_labels = ['Fresh', 'Rotten']

# ----------------------------
# 7) Prediction wrapper for Gradio
# ----------------------------
UNKNOWN_THRESH = 0.70

def annotate_image_with_text(pil_img, lines):
    """Return a copy of pil_img with text lines anchored at top-left."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a default truetype font if available
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    x, y = 10, 10
    for line in lines:
        draw.text((x, y), line, fill=(255,255,255), font=font)
        y += 22
    return img

def predict(image: Image.Image):
    # Save a temp copy for hsv processing (OpenCV reads from path)
    tmp_fp = os.path.join(tempfile.gettempdir(), f"fruit_{uuid.uuid4().hex}.jpg")
    image.save(tmp_fp)

    # Face/human detection
    if detect_human_face(image):
        return {"error": "Image appears to contain a human face. Please upload only fruit images."}, image

    # Preprocess and move to device
    img_tensor = fruit_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fruit_pred, fresh_pred, shelf_pred = model(img_tensor)

    # fruit_pred shape: [1,9] -> softmax probabilities
    probs = torch.softmax(fruit_pred, dim=1)[0].cpu().numpy()
    max_idx = int(np.argmax(probs))
    max_prob = float(probs[max_idx])

    predicted_fruit = fruit_labels[max_idx]
    predicted_freshness = fresh_labels[int(torch.argmax(fresh_pred, dim=1).item())]
    raw_shelf_output = float(shelf_pred.item())

    # Unknown detection
    if max_prob < UNKNOWN_THRESH:
        return {"error": "Model is not confident (max prob {:.3f}). This may be an unsupported fruit or poor image.".format(max_prob)}, image

    # Get shelf life and method
    shelf_life_days, method = hybrid_shelf_life(predicted_fruit, predicted_freshness, raw_shelf_output, tmp_fp)

    # Prepare JSON result
    result = {
        "Fruit": predicted_fruit,
        "Fruit Probability": round(max_prob, 4),
        "Freshness": predicted_freshness,
        "Estimated Shelf Life (days)": shelf_life_days,
        "Method Used": method,
        "Raw Regression Output": round(raw_shelf_output, 4)
    }

    # Annotate image for display
    lines = [
        f"Fruit: {predicted_fruit} ({max_prob:.2f})",
        f"Freshness: {predicted_freshness}",
        f"Shelf-life (days): {shelf_life_days}  [{method}]"
    ]
    annotated = annotate_image_with_text(image, lines)

    # cleanup temp
    try:
        os.remove(tmp_fp)
    except Exception:
        pass

    return result, annotated

# ----------------------------
# 8) Gradio interface
# ----------------------------
title = "Fruit Freshness & Shelf-Life Predictor"
description = (
    "Upload a fruit/vegetable image. The app will detect if a human face is present (rejects such images), "
    "classify the fruit (9 classes), predict fresh/rotten, and estimate shelf-life using HSV for supported fruits "
    "or the regression head otherwise."
)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload fruit image"),
    outputs=[gr.JSON(label="Prediction"), gr.Image(type="pil", label="Annotated Image")],
    title=title,
    description=description,
    allow_flagging="never",
    examples=None
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


