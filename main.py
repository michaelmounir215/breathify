from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import io
import os

# إنشاء التطبيق
app = FastAPI()

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ربط ملفات الواجهة الثابتة
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# صفحة البداية
@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return f.read()
    return "<h1>API is running, but index.html not found.</h1>"

# أسماء الأمراض
diseases = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
    "Fibrosis", "Pleural Thickening", "Hernia", "Fracture"
]

# تحميل الموديل المدرب
model = models.densenet121(weights=None)
model.classifier = nn.Linear(1024, 15)

# تأكد من وجود الملف
model_path = "chest_xray_chexnet.pth"
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# تجهيز الصورة قبل التنبؤ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# API للتشخيص
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)

        probs = output[0].tolist()
        max_idx = probs.index(max(probs))
        top_disease = diseases[max_idx]
        top_probability = round(probs[max_idx], 4)

        return {
            "top_disease": top_disease,
            "probability": top_probability
        }

    except Exception as e:
        return {"error": str(e)}
