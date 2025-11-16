from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from deep_translator import GoogleTranslator
from gtts import gTTS
import time
import hashlib
import traceback
from typing import Optional
from pathlib import Path

# -----------------------------
# CONSTANTS
# -----------------------------
MODEL_NAME = "microsoft/trocr-base-printed"
OUTPUT_DIR = Path("outputs")
MAX_TEXT_LENGTH = 20000
MIN_IMAGE_WIDTH = 600
MAX_IMAGE_WIDTH = 1600
CONTRAST_FACTOR = 1.08
MAX_LENGTH = 128
NUM_BEAMS = 2

# Ensure outputs folder exists
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# GLOBALS
# -----------------------------
processor: Optional[TrOCRProcessor] = None
model: Optional[VisionEncoderDecoderModel] = None
device: Optional[str] = None
models_loaded: bool = False
models_loading: bool = False


# -----------------------------
# MODEL LOADING (LAZY)
# -----------------------------
def load_models() -> bool:
    """
    Load TrOCR processor and model on first request.
    
    Uses a simple thread-safe lock mechanism to prevent multiple simultaneous loads.
    The model is loaded lazily to reduce startup time.
    
    Returns:
        bool: True if models loaded successfully, False otherwise
    """
    global processor, model, device, models_loaded, models_loading

    if models_loaded:
        return True

    if models_loading:
        # Wait until another request finishes loading
        while models_loading:
            time.sleep(0.5)
        return models_loaded

    models_loading = True
    try:
        print("🚀 Loading TrOCR model (this may take a while)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Using device: {device}")

        # Load processor and model
        processor_local = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model_local = VisionEncoderDecoderModel.from_pretrained(
            MODEL_NAME,
            ignore_mismatched_sizes=True
        ).to(device)
        model_local.eval()

        # Apply to globals
        globals_map = globals()
        globals_map["processor"] = processor_local
        globals_map["model"] = model_local
        globals_map["device"] = device

        # CPU optimization
        if device == "cpu":
            torch.set_num_threads(4)

        models_loaded = True
        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

    finally:
        models_loading = False


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img_rgb: np.ndarray) -> Image.Image:
    """
    Preprocess image for TrOCR recognition.
    
    Resizes image to optimal dimensions and enhances contrast for better OCR accuracy.
    
    Args:
        img_rgb: Input image as RGB numpy array
        
    Returns:
        PIL Image: Preprocessed image ready for OCR
    """
    img = Image.fromarray(img_rgb).convert("RGB")
    w, h = img.size

    # Resize to optimal width range for sentence reading
    if w < MIN_IMAGE_WIDTH:
        scale = MIN_IMAGE_WIDTH / w
        img = img.resize((MIN_IMAGE_WIDTH, int(h * scale)), Image.Resampling.LANCZOS)
    elif w > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / w
        img = img.resize((MAX_IMAGE_WIDTH, int(h * scale)), Image.Resampling.LANCZOS)

    # Enhance contrast for better text recognition
    img = ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)
    return img


# -----------------------------
# TrOCR RECOGNITION
# -----------------------------
def recognize_with_trocr(img_rgb: np.ndarray) -> str:
    """
    Run TrOCR text recognition on an RGB image.
    
    Args:
        img_rgb: Input image as RGB numpy array
        
    Returns:
        str: Extracted text from the image
        
    Raises:
        Exception: If model fails to load
    """
    if not load_models():
        raise Exception("Model failed to load on server.")

    img = preprocess_image(img_rgb)

    with torch.no_grad():
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(
            pixel_values,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=False,
            no_repeat_ngram_size=2,
            length_penalty=0.0,
            do_sample=False
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(
    title="AI Handwritten OCR",
    description="Modern OCR application with translation and text-to-speech capabilities",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# ROOT HTML (Animated QuickLearn.ai-like UI)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Handwriting OCR — Premium Experience</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg1:#0a0e27; --bg2:#0f1629;
  --card:#0d1222; --glass: rgba(255,255,255,0.04);
  --accent1:#8b5cf6; --accent2:#3b82f6; --accent3:#06b6d4;
  --muted:#94a3b8; --text:#e2e8f0;
  --glass-border: rgba(255,255,255,0.06);
  --success:#10b981; --error:#ef4444;
  --shadow-glow: 0 0 40px rgba(139,92,246,0.15);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  margin:0; 
  font-family:'Inter',system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  background: 
    radial-gradient(ellipse 1200px 600px at 20% 0%, rgba(59,130,246,0.08), transparent 50%),
    radial-gradient(ellipse 1000px 500px at 80% 100%, rgba(139,92,246,0.08), transparent 50%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
  background-attachment: fixed;
  color:var(--text);
  -webkit-font-smoothing:antialiased; 
  -moz-osx-font-smoothing:grayscale;
  overflow-y:auto;
  animation: pageFade .8s cubic-bezier(0.4,0,0.2,1) both;
  position:relative;
}
body::before{
  content:'';position:fixed;inset:0;background:
    radial-gradient(circle at 20% 30%, rgba(139,92,246,0.03) 0%, transparent 50%),
    radial-gradient(circle at 80% 70%, rgba(59,130,246,0.03) 0%, transparent 50%);
  pointer-events:none;z-index:0;
}

@keyframes pageFade { 
  from {opacity:0; transform: translateY(12px)} 
  to {opacity:1; transform: none} 
}

/* NAV */
.nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:20px 40px;position:sticky;top:0;
  background:rgba(10,14,39,0.7);
  backdrop-filter: blur(20px) saturate(180%);
  border-bottom:1px solid var(--glass-border);
  z-index:100;
  box-shadow:0 4px 24px rgba(0,0,0,0.2);
}
.brand{
  font-weight:900;color:var(--accent1);font-size:22px;
  letter-spacing:-0.5px;display:flex;align-items:center;gap:14px;
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  -webkit-background-clip:text;color:transparent;
}
.logo-dot{
  width:12px;height:12px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  box-shadow:0 0 20px rgba(139,92,246,0.5);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100%{opacity:1;transform:scale(1)}
  50%{opacity:0.8;transform:scale(1.1)}
}
.nav-right{display:flex;gap:12px;align-items:center}
.badge{
  padding:8px 14px;border-radius:12px;
  background:linear-gradient(135deg, rgba(139,92,246,0.12), rgba(59,130,246,0.08));
  color:var(--accent2);font-weight:700;font-size:13px;
  border:1px solid rgba(139,92,246,0.2);
  box-shadow:0 4px 12px rgba(139,92,246,0.1);
}

/* layout */
.container{max-width:1280px;margin:60px auto;padding:0 32px;position:relative;z-index:1}
.header{text-align:center;margin-bottom:48px}
h1{
  font-size:56px;margin:12px 0;font-weight:900;letter-spacing:-1.5px;
  background:linear-gradient(135deg,#e0e7ff 0%,#c7d2fe 50%,#a5b4fc 100%);
  -webkit-background-clip:text;color:transparent;
  line-height:1.1;
}
.lead{
  color:var(--muted);max-width:720px;margin:20px auto 0;
  font-size:17px;line-height:1.7;font-weight:400;
}

/* grid */
.grid{
  display:grid;grid-template-columns:1fr 460px;
  gap:32px;align-items:start;margin-top:40px;
}
@media(max-width:1024px){.grid{grid-template-columns:1fr}}

/* card */
.card{
  border-radius:20px;padding:32px;
  background:linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
  border:1px solid var(--glass-border);
  box-shadow:0 24px 64px rgba(0,0,0,0.4), var(--shadow-glow);
  backdrop-filter:blur(10px);
  transition:transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover{transform:translateY(-4px);box-shadow:0 32px 80px rgba(0,0,0,0.5), 0 0 60px rgba(139,92,246,0.2);}

/* upload wrapper */
.upload-wrapper{
  position:relative;
  border-radius:18px;
}

/* upload area */
.upload{
  border-radius:18px;padding:48px 32px;text-align:center;
  cursor:pointer;border:2.5px dashed rgba(255,255,255,0.08);
  transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
  position:relative;overflow:hidden;
  background:linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%);
  z-index:1;
}
.upload:hover, .upload.drag-over{
  transform:translateY(-8px);
  box-shadow:0 40px 100px rgba(139,92,246,0.15);
  border-color:rgba(139,92,246,0.3);
  background:linear-gradient(180deg, rgba(139,92,246,0.05) 0%, rgba(59,130,246,0.02) 100%);
}
.upload.processing{
  border:2.5px solid rgba(139,92,246,0.6);
  box-shadow:0 0 30px rgba(139,92,246,0.5),
             0 0 60px rgba(139,92,246,0.4),
             0 0 90px rgba(139,92,246,0.3),
             0 0 120px rgba(59,130,246,0.2),
             inset 0 0 30px rgba(139,92,246,0.1);
  animation:boxGlow 2s ease-in-out infinite;
  background:linear-gradient(180deg, rgba(139,92,246,0.08) 0%, rgba(59,130,246,0.04) 100%);
}
@keyframes boxGlow{
  0%,100%{
    box-shadow:0 0 30px rgba(139,92,246,0.5),
               0 0 60px rgba(139,92,246,0.4),
               0 0 90px rgba(139,92,246,0.3),
               0 0 120px rgba(59,130,246,0.2),
               inset 0 0 30px rgba(139,92,246,0.1);
  }
  50%{
    box-shadow:0 0 40px rgba(139,92,246,0.7),
               0 0 80px rgba(139,92,246,0.6),
               0 0 120px rgba(139,92,246,0.5),
               0 0 160px rgba(59,130,246,0.3),
               inset 0 0 40px rgba(139,92,246,0.15);
  }
}
.upload .icon{
  font-size:64px;margin-bottom:20px;display:inline-block;
  filter:drop-shadow(0 8px 24px rgba(59,130,246,0.2));
  transition:transform 0.3s ease;
}
.upload:hover .icon{transform:scale(1.1) rotate(5deg);}
.upload h3{
  font-size:24px;margin:8px 0;color:#f1f5f9;
  font-weight:700;letter-spacing:-0.5px;
}
.upload p{
  color:var(--muted);margin:12px 0 24px;
  font-size:15px;line-height:1.6;
}

/* buttons */
.file-btn, .action-btn{
  padding:14px 28px;border-radius:12px;
  font-weight:700;font-size:15px;cursor:pointer;
  transition:all 0.2s ease;border:none;
  font-family:inherit;
}
.file-btn{
  background:linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.1));
  color:var(--accent2);border:1.5px solid rgba(139,92,246,0.3);
  box-shadow:0 4px 16px rgba(139,92,246,0.1);
}
.file-btn:hover{
  background:linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.15));
  transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(139,92,246,0.2);
}
.file-btn:active{transform:translateY(0);}
.action-btn{
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  color:#ffffff;box-shadow:0 8px 24px rgba(139,92,246,0.3);
  white-space:nowrap;
}
.action-btn:hover{
  transform:translateY(-2px);
  box-shadow:0 12px 32px rgba(139,92,246,0.4);
}
.action-btn:active{transform:translateY(0);}
.action-btn:disabled{
  opacity:0.6;cursor:not-allowed;
  transform:none !important;
}

/* floating animation */
.floating{animation:floaty 6s ease-in-out infinite;}
@keyframes floaty { 
  0%,100%{transform:translateY(0) rotate(0deg)} 
  50%{transform:translateY(-12px) rotate(2deg)} 
}

/* animated gradient title */
.glow-title{
  display:inline-block;
  background:linear-gradient(90deg, #a5b4fc, #c7d2fe, #e0e7ff, #c7d2fe, #a5b4fc);
  background-size:300% 100%;
  -webkit-background-clip:text;color:transparent;
  animation:gradientShift 8s linear infinite;
}
@keyframes gradientShift { 
  0%{background-position:0% 0%}
  100%{background-position:300% 0%} 
}

/* scan ring */
.scan-ring{
  position:absolute;inset:-3px;border-radius:18px;
  pointer-events:none;
  background:conic-gradient(from 0deg, transparent, rgba(139,92,246,0.2), transparent, rgba(59,130,246,0.2), transparent);
  opacity:0;transform:scale(0.95);
  transition:opacity 0.4s ease, transform 0.4s ease;
  animation:rotate 4s linear infinite;
}
@keyframes rotate{to{transform:rotate(360deg) scale(0.95);}}
.upload:hover .scan-ring{opacity:1;filter:blur(12px);}


/* badges */
.meta{display:flex;gap:12px;justify-content:center;margin-top:20px;flex-wrap:wrap;}
.pill{
  background:rgba(255,255,255,0.04);
  color:var(--muted);padding:10px 16px;
  border-radius:12px;font-weight:600;font-size:13px;
  border:1px solid rgba(255,255,255,0.06);
  transition:all 0.2s ease;
}
.pill:hover{
  background:rgba(255,255,255,0.08);
  border-color:rgba(139,92,246,0.3);
  transform:translateY(-2px);
}

/* preview */
.side .preview{text-align:center;margin-bottom:20px;}
.preview img{
  max-width:100%;border-radius:16px;
  box-shadow:0 24px 64px rgba(0,0,0,0.5);
  border:2px solid rgba(255,255,255,0.08);
  transition:transform 0.3s ease;
}
.preview img:hover{transform:scale(1.02);}

/* panels */
.panel{
  margin-bottom:20px;padding:20px;
  border-radius:14px;
  background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.06);
  transition:all 0.3s ease;
}
.panel:hover{border-color:rgba(139,92,246,0.2);}
.p-title{
  font-weight:800;color:#e2e8f0;margin-bottom:12px;
  font-size:16px;letter-spacing:-0.3px;
  display:flex;align-items:center;gap:8px;
}
.p-title::before{
  content:'';width:4px;height:16px;
  background:linear-gradient(135deg,var(--accent1),var(--accent2));
  border-radius:2px;
}
.text-box{
  min-height:160px;padding:16px;
  border-radius:12px;
  background:linear-gradient(180deg, rgba(2,6,23,0.6), rgba(2,6,23,0.8));
  color:#dbeafe;border:1px solid rgba(255,255,255,0.04);
  overflow:auto;white-space:pre-wrap;
  font-size:15px;line-height:1.7;
  font-family:'Inter',sans-serif;
}
.text-box:empty::before{
  content:'Waiting for content...';
  color:var(--muted);font-style:italic;
}

/* controls */
.controls{
  display:flex;gap:12px;align-items:center;
  margin-top:16px;flex-wrap:wrap;
}

/* Custom Dropdown */
.custom-select-wrapper{
  position:relative;flex:1;min-width:200px;
}
.custom-select{
  position:relative;cursor:pointer;
}
.custom-select-trigger{
  padding:14px 48px 14px 16px;border-radius:12px;
  background:linear-gradient(135deg, rgba(139,92,246,0.1), rgba(59,130,246,0.08));
  color:#dcecff;border:1.5px solid rgba(139,92,246,0.3);
  font-family:inherit;font-size:14px;font-weight:600;
  transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 4px 12px rgba(139,92,246,0.1);
  position:relative;overflow:hidden;
}
.custom-select-trigger::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.1));
  opacity:0;transition:opacity 0.3s ease;
}
.custom-select-trigger:hover::before{opacity:1;}
.custom-select-trigger:hover{
  border-color:rgba(139,92,246,0.5);
  box-shadow:0 6px 20px rgba(139,92,246,0.2);
  transform:translateY(-2px);
}
.custom-select-trigger.active{
  border-color:var(--accent1);
  box-shadow:0 0 0 3px rgba(139,92,246,0.2), 0 8px 24px rgba(139,92,246,0.3);
}
.custom-select-trigger span{
  position:relative;z-index:1;
  display:flex;align-items:center;gap:8px;
}
.custom-select-arrow{
  position:relative;z-index:1;
  width:20px;height:20px;
  transition:transform 0.3s ease;
  display:flex;align-items:center;justify-content:center;
}
.custom-select-arrow::before{
  content:'▼';font-size:10px;color:var(--accent2);
  transition:transform 0.3s ease;
}
.custom-select.active .custom-select-arrow::before{
  transform:rotate(180deg);
}
.custom-select-options{
  position:absolute;top:calc(100% + 8px);left:0;right:0;
  background:rgba(13,18,34,0.98);
  backdrop-filter:blur(20px) saturate(180%);
  border:1.5px solid rgba(139,92,246,0.3);
  border-radius:12px;
  box-shadow:0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(139,92,246,0.2);
  max-height:300px;overflow-y:auto;
  z-index:1000;
  opacity:0;visibility:hidden;transform:translateY(-10px);
  transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
  padding:8px;
}
.custom-select-options::-webkit-scrollbar{
  width:6px;
}
.custom-select-options::-webkit-scrollbar-track{
  background:rgba(255,255,255,0.02);border-radius:3px;
}
.custom-select-options::-webkit-scrollbar-thumb{
  background:rgba(139,92,246,0.3);border-radius:3px;
}
.custom-select-options::-webkit-scrollbar-thumb:hover{
  background:rgba(139,92,246,0.5);
}
.custom-select.active .custom-select-options{
  opacity:1;visibility:visible;transform:translateY(0);
}
.custom-select-option{
  padding:12px 16px;border-radius:8px;
  color:#dcecff;font-size:14px;font-weight:500;
  cursor:pointer;transition:all 0.2s ease;
  display:flex;align-items:center;gap:10px;
  margin-bottom:4px;
  position:relative;
}
.custom-select-option::before{
  content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:3px;height:0;background:linear-gradient(135deg,var(--accent1),var(--accent2));
  border-radius:0 2px 2px 0;
  transition:height 0.2s ease;
}
.custom-select-option:hover{
  background:linear-gradient(90deg, rgba(139,92,246,0.15), rgba(59,130,246,0.1));
  transform:translateX(4px);
  padding-left:20px;
}
.custom-select-option:hover::before{height:60%;}
.custom-select-option.selected{
  background:linear-gradient(90deg, rgba(139,92,246,0.2), rgba(59,130,246,0.15));
  color:#ffffff;font-weight:600;
  padding-left:20px;
}
.custom-select-option.selected::before{height:60%;}
.custom-select-option .flag{font-size:18px;}

/* Native select hidden */
select#targetLang{
  display:none;
}

/* loader */
.loader{
  width:48px;height:48px;border-radius:50%;
  display:inline-block;border:4px solid rgba(255,255,255,0.1);
  border-top-color:var(--accent2);
  animation:spin 0.8s linear infinite;
  margin:16px auto;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* shimmer */
.shimmer{
  position:relative;overflow:hidden;
  background:linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.06), rgba(255,255,255,0.03));
}
.shimmer::after{
  content:'';position:absolute;left:-150%;top:0;
  height:100%;width:150%;
  background:linear-gradient(90deg,transparent, rgba(255,255,255,0.1), transparent);
  animation:shimmer 2s linear infinite;
}
@keyframes shimmer{100%{left:150%}}

/* audio */
audio{
  width:100%;margin-top:16px;border-radius:10px;
  background:rgba(255,255,255,0.02);
}

/* footer */
.footer{
  text-align:center;color:var(--muted);
  margin-top:60px;font-size:14px;
  padding:24px 0;
}

/* notifications */
.notification{
  position:fixed;top:100px;right:32px;
  padding:16px 24px;border-radius:12px;
  background:rgba(10,14,39,0.95);
  border:1px solid var(--glass-border);
  backdrop-filter:blur(20px);
  box-shadow:0 12px 40px rgba(0,0,0,0.4);
  z-index:1000;
  animation:slideIn 0.3s ease;
  max-width:400px;
}
@keyframes slideIn{
  from{transform:translateX(400px);opacity:0;}
  to{transform:translateX(0);opacity:1;}
}
.notification.error{border-left:4px solid var(--error);}
.notification.success{border-left:4px solid var(--success);}

/* responsive */
@media(max-width:768px){
  .nav{padding:16px 20px}
  .container{padding:0 20px;margin:40px auto}
  .upload{padding:32px 20px}
  h1{font-size:36px}
  .lead{font-size:15px}
  .grid{gap:24px}
  .card{padding:24px}
}
@media(max-width:480px){
  h1{font-size:28px}
  .upload h3{font-size:20px}
  .controls{flex-direction:column}
  .custom-select-wrapper{width:100%}
}
</style>
</head>
<body>
  <nav class="nav">
    <div class="brand">
      <span class="logo-dot"></span>
      <span>AI OCR</span>
    </div>
    <div class="nav-right">
      <div class="badge">TrOCR • FastAPI</div>
    </div>
  </nav>

  <div class="container">
    <div class="header">
      <h1><span class="glow-title">Akshar-The Handwritten</span> Text Recognition App</h1>
      <p class="lead">Upload an image — our TrOCR-powered AI reads handwritten or printed text, translates it to any language, and generates natural speech. First load may take a few seconds while the model initializes.</p>
    </div>

    <div class="grid">
      <div>
        <div class="card upload-card">
          <div class="upload-wrapper" id="uploadWrapper">
            <div class="upload" id="uploadBox">
              <div class="scan-ring"></div>
              <div class="icon floating">📸</div>
            <h3>Drop or choose an image</h3>
            <p>Works with printed and handwritten images. Try clear, well-lit photos for best results. Supports PNG, JPG, and JPEG formats.</p>
            <label class="file-btn" for="imageInput">Choose File</label>
            <input type="file" id="imageInput" accept="image/*" style="display:none">
            <div class="meta">
              <div class="pill">TrOCR AI</div>
              <div class="pill">Multi-Language</div>
              <div class="pill">Text-to-Speech</div>
            </div>
            </div>
          </div>

          <div style="text-align:center;margin-top:24px">
            <button class="action-btn" id="recognizeBtn" onclick="recognizeText()">
              <span>Recognize Text</span>
            </button>
          </div>
        </div>

        <div style="height:24px"></div>

        <div class="card">
          <div class="p-title">Tips & Notes</div>
          <div style="color:var(--muted);line-height:1.8;font-size:14px">
            • Model loads on first request — this may take 10-45s depending on your machine.<br>
            • For offline TTS, swap gTTS for another engine. Audio files are saved in the server outputs/ folder.<br>
            • Use PNG/JPG images. Large images are automatically resized for optimal processing.<br>
            • Best results with high-contrast, well-lit images containing clear text.
          </div>
        </div>
      </div>

      <div class="side">
        <div class="card">
          <div class="preview">
            <img id="uploadedImage" src="" alt="Uploaded preview" style="display:none"/>
          </div>

          <div class="panel">
            <div class="p-title">Recognized Text</div>
            <div id="recognizedText" class="text-box">No text yet — upload an image and press Recognize.</div>
          </div>

          <div class="panel">
            <div class="p-title">Translate & Speech</div>
            <div class="controls">
              <div class="custom-select-wrapper">
                <div class="custom-select" id="customSelect">
                  <div class="custom-select-trigger" id="selectTrigger">
                    <span><span class="flag">🇺🇸</span> English</span>
                    <div class="custom-select-arrow"></div>
                  </div>
                  <div class="custom-select-options" id="selectOptions">
                    <div class="custom-select-option selected" data-value="en">
                      <span class="flag">🇺🇸</span> English
                    </div>
                    <div class="custom-select-option" data-value="hi">
                      <span class="flag">🇮🇳</span> Hindi
                    </div>
                    <div class="custom-select-option" data-value="es">
                      <span class="flag">🇪🇸</span> Spanish
                    </div>
                    <div class="custom-select-option" data-value="fr">
                      <span class="flag">🇫🇷</span> French
                    </div>
                    <div class="custom-select-option" data-value="de">
                      <span class="flag">🇩🇪</span> German
                    </div>
                    <div class="custom-select-option" data-value="zh-cn">
                      <span class="flag">🇨🇳</span> Chinese (Simplified)
                    </div>
                    <div class="custom-select-option" data-value="ja">
                      <span class="flag">🇯🇵</span> Japanese
                    </div>
                    <div class="custom-select-option" data-value="ar">
                      <span class="flag">🇸🇦</span> Arabic
                    </div>
                    <div class="custom-select-option" data-value="pt">
                      <span class="flag">🇵🇹</span> Portuguese
                    </div>
                    <div class="custom-select-option" data-value="ru">
                      <span class="flag">🇷🇺</span> Russian
                    </div>
                    <div class="custom-select-option" data-value="it">
                      <span class="flag">🇮🇹</span> Italian
                    </div>
                    <div class="custom-select-option" data-value="ko">
                      <span class="flag">🇰🇷</span> Korean
                    </div>
                  </div>
                </div>
                <select id="targetLang" style="display:none">
                  <option value="en" selected>English</option>
                  <option value="hi">Hindi</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                  <option value="zh-cn">Chinese (Simplified)</option>
                  <option value="ja">Japanese</option>
                  <option value="ar">Arabic</option>
                  <option value="pt">Portuguese</option>
                  <option value="ru">Russian</option>
                  <option value="it">Italian</option>
                  <option value="ko">Korean</option>
                </select>
              </div>
              <button class="action-btn" id="translateBtn" onclick="translateText()">
                <span>Translate & Speak</span>
              </button>
            </div>
            <div id="translatedText" class="text-box" style="min-height:140px;margin-top:16px"></div>
            <audio id="audioPlayer" controls style="display:none"></audio>
          </div>
        </div>
      </div>
    </div>

    <div class="footer">
      Made using TrOCR • FastAPI • Modern AI Technology
    </div>
  </div>

<script>
let currentText = "";

// Custom Dropdown Functionality
const customSelect = document.getElementById('customSelect');
const selectTrigger = document.getElementById('selectTrigger');
const selectOptions = document.getElementById('selectOptions');
const nativeSelect = document.getElementById('targetLang');
const options = selectOptions.querySelectorAll('.custom-select-option');

// Toggle dropdown
selectTrigger.addEventListener('click', (e) => {
  e.stopPropagation();
  customSelect.classList.toggle('active');
  selectTrigger.classList.toggle('active');
});

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
  if(!customSelect.contains(e.target)){
    customSelect.classList.remove('active');
    selectTrigger.classList.remove('active');
  }
});

// Handle option selection
options.forEach(option => {
  option.addEventListener('click', () => {
    const value = option.dataset.value;
    const text = option.textContent.trim();
    
    // Update native select
    nativeSelect.value = value;
    
    // Update trigger text
    selectTrigger.querySelector('span').innerHTML = option.innerHTML;
    
    // Update selected state
    options.forEach(opt => opt.classList.remove('selected'));
    option.classList.add('selected');
    
    // Close dropdown
    customSelect.classList.remove('active');
    selectTrigger.classList.remove('active');
  });
});

// File input handler
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');

imageInput.addEventListener('change', function(e){
  if(e.target.files.length > 0){
    const imgFile = e.target.files[0];
    const url = URL.createObjectURL(imgFile);
    const imgEl = document.getElementById('uploadedImage');
    imgEl.src = url;
    imgEl.style.display = 'block';
    uploadBox.classList.remove('drag-over');
  }
});

// Drag and drop
uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('drag-over');
});
uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('drag-over');
});
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('drag-over');
  if(e.dataTransfer.files.length > 0){
    imageInput.files = e.dataTransfer.files;
    const event = new Event('change', {bubbles: true});
    imageInput.dispatchEvent(event);
  }
});

// Utility functions
function showNotification(message, type = 'info'){
  const notif = document.createElement('div');
  notif.className = `notification ${type}`;
  notif.textContent = message;
  document.body.appendChild(notif);
  setTimeout(() => {
    notif.style.animation = 'slideIn 0.3s ease reverse';
    setTimeout(() => notif.remove(), 300);
  }, 3000);
}

// Recognize text
async function recognizeText(){
  const fileInput = document.getElementById('imageInput');
  if(!fileInput.files || fileInput.files.length === 0){
    showNotification('Please upload an image first.', 'error');
    return;
  }
  
  const form = new FormData();
  form.append('file', fileInput.files[0]);

  const recEl = document.getElementById('recognizedText');
  const old = recEl.textContent;
  recEl.textContent = 'Processing... (model may load on first request)';
  recEl.classList.add('shimmer');

  const btn = document.getElementById('recognizeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span>Processing...</span>';
  
  // Add glowing effect to upload box
  uploadBox.classList.add('processing');

  try {
    const resp = await fetch('/recognize', { method:'POST', body: form });
    if(!resp.ok){
      const errorData = await resp.json().catch(() => ({detail: 'Unknown error'}));
      throw new Error(errorData.detail || `Error ${resp.status}`);
    }
    const data = await resp.json();
    currentText = data.text || '';
    recEl.textContent = currentText || 'No text found in the image.';
    recEl.classList.remove('shimmer');
    if(currentText) showNotification('Text recognized successfully!', 'success');
  } catch(err) {
    console.error(err);
    showNotification('Error: ' + (err.message || err), 'error');
    recEl.textContent = old;
    recEl.classList.remove('shimmer');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>Recognize Text</span>';
    // Remove glowing effect from upload box
    uploadBox.classList.remove('processing');
  }
}

// Translate text
async function translateText(){
  if(!currentText){
    showNotification('Please recognize text first.', 'error');
    return;
  }
  
  const lang = document.getElementById('targetLang').value;
  const form = new FormData();
  form.append('text', currentText);
  form.append('target_lang', lang);

  const transEl = document.getElementById('translatedText');
  transEl.textContent = 'Translating...';
  transEl.classList.add('shimmer');

  const btn = document.getElementById('translateBtn');
  btn.disabled = true;
  btn.innerHTML = '<span>Translating...</span>';

  try {
    const resp = await fetch('/translate', { method:'POST', body: form });
    if(!resp.ok){
      const errorData = await resp.json().catch(() => ({detail: 'Translation failed'}));
      throw new Error(errorData.detail || 'Translation failed');
    }
    const data = await resp.json();
    transEl.textContent = data.translated_text || '';
    transEl.classList.remove('shimmer');
    
    if(data.audio_url){
      const audio = document.getElementById('audioPlayer');
      audio.src = data.audio_url;
      audio.style.display = 'block';
      audio.play().catch(() => {
        // Autoplay may be blocked, user can press play
      });
      showNotification('Translation and audio generated!', 'success');
    }
  } catch(err){
    console.error(err);
    showNotification('Error: ' + (err.message || err), 'error');
    transEl.textContent = '';
    transEl.classList.remove('shimmer');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>Translate & Speak</span>';
  }
}
</script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


# -----------------------------
# RECOGNIZE ENDPOINT
# -----------------------------
@app.post("/recognize")
async def recognize_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Recognize text from an uploaded image.
    
    Args:
        file: Image file to process
        
    Returns:
        JSONResponse: Extracted text and status
        
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = recognize_with_trocr(img_rgb)

        if not text:
            raise HTTPException(status_code=400, detail="No text detected in image")

        return JSONResponse({"text": text, "status": "success"})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# -----------------------------
# TRANSLATE & AUDIO ENDPOINT
# -----------------------------
@app.post("/translate")
async def translate_text(text: str = Form(...), target_lang: str = Form("en")) -> dict:
    """
    Translate text and generate audio speech.
    
    Args:
        text: Text to translate
        target_lang: Target language code (default: "en")
        
    Returns:
        dict: Translated text and audio URL
        
    Raises:
        HTTPException: If translation fails or text is too long
    """
    try:
        # Validate text length
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters."
            )
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Translate text
        translator = GoogleTranslator(source="auto", target=target_lang)
        translated = translator.translate(text)

        # Generate stable filename using hash + timestamp
        text_hash = hashlib.sha256(f"{translated}_{target_lang}".encode("utf-8")).hexdigest()[:16]
        filename = OUTPUT_DIR / f"{text_hash}_{int(time.time())}.mp3"

        # Generate TTS audio
        tts = gTTS(text=translated, lang=target_lang)
        tts.save(str(filename))

        # Return audio URL
        audio_url = f"/audio/{filename.name}"
        return {"translated_text": translated, "audio_url": audio_url}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


# -----------------------------
# SERVE AUDIO FILES
# -----------------------------
@app.get("/audio/{filename}")
async def serve_audio(filename: str) -> FileResponse:
    """
    Serve generated audio files.
    
    Args:
        filename: Name of the audio file
        
    Returns:
        FileResponse: Audio file
        
    Raises:
        HTTPException: If file not found
    """
    # Security: prevent directory traversal
    safe_filename = os.path.basename(filename)
    path = OUTPUT_DIR / safe_filename
    
    if path.exists() and path.is_file():
        return FileResponse(path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
async def health() -> dict:
    """
    Health check endpoint.
    
    Returns:
        dict: Server status and model information
    """
    return {
        "status": "running",
        "device": device or "unknown",
        "model_loaded": models_loaded,
        "model_name": MODEL_NAME
    }


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    # run via Uvicorn
    import uvicorn
    uvicorn.run("main1:app", host="0.0.0.0", port=8000, reload=False)
