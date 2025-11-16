#!/usr/bin/env python
# coding: utf-8

# ============================================================================================
# HANDWRITTEN / CLEAN-WRITING TEXT RECOGNITION — WITH FULL PREPROCESSING
# USING TrOCR ONLY (NO EASYOCR)
# ============================================================================================

# -----------------------------
# IMPORTS
# -----------------------------
import cv2
import torch
import numpy as np
import os
import sys
from PIL import Image

# Optional imports with error handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be skipped.")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    print("Error: transformers library not found. Please install: pip install transformers")
    sys.exit(1)

try:
    from googletrans import Translator
except ImportError:
    print("Error: googletrans library not found. Please install: pip install googletrans==4.0.0-rc1")
    sys.exit(1)

try:
    from gtts import gTTS
except ImportError:
    print("Error: gTTS library not found. Please install: pip install gtts")
    sys.exit(1)

# -----------------------------
# DEVICE CHECK
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# LOAD TrOCR MODEL
# -----------------------------
print("Loading TrOCR model (this may take a while)...")
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed",
        ignore_mismatched_sizes=True
    ).to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading TrOCR model: {e}")
    print("Please ensure you have internet connection for first-time download.")
    sys.exit(1)

try:
    translator = Translator()
except Exception as e:
    print(f"Warning: Could not initialize translator: {e}")
    translator = None

# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image_opencv(image_path):
    """
    Enhanced preprocessing pipeline for better OCR accuracy.
    Includes: denoising, binarization, contrast enhancement, and sharpening.
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}. Please check the file path.")

    # Handle both color and grayscale images
    if len(img.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Denoise (Gaussian Blur) - reduces noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Additional denoising using Non-local Means (optional, slower but better)
    # denoised = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)

    # Morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

    # Adaptive Threshold (binarization) - better for varying lighting
    thresh = cv2.adaptiveThreshold(
        cleaned, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    # Contrast Enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(thresh)

    # Additional sharpening for better text clarity
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # Resize (maintain aspect ratio) - TrOCR works better with larger images
    h, w = sharpened.shape
    scale = 1024 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(sharpened, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Convert to PIL image (TrOCR needs RGB)
    pil_img = Image.fromarray(resized).convert("RGB")

    return pil_img, enhanced, resized


# -----------------------------
# OCR RECOGNITION FUNCTION
# -----------------------------
def recognize_image(img_pil):
    pixel_values = processor(img_pil, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# -----------------------------
# SET IMAGE PATH
# -----------------------------
# Change this to your image path
IMAGE_PATH = "sample.jpg"  # Default local path
# Or specify as command line argument: python sgp_sem_5.py path/to/image.jpg
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found: {IMAGE_PATH}")
    print("Please provide a valid image path.")
    sys.exit(1)

print("Reading:", IMAGE_PATH)

# -----------------------------
# RUN PREPROCESSING
# -----------------------------
print("\n Preprocessing image...")
try:
    pre_img, thresh_img, resized_img = preprocess_image_opencv(IMAGE_PATH)
    print(" Preprocessing complete! (Grayscale, Denoising, Thresholding, CLAHE, Sharpening, Resizing)")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    sys.exit(1)

# -----------------------------
# SHOW THE PREPROCESSING STAGES
# -----------------------------
if MATPLOTLIB_AVAILABLE:
    try:
        plt.figure(figsize=(15, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(thresh_img, cmap='gray')
        plt.title("Threshold + CLAHE", fontsize=14)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(resized_img, cmap='gray')
        plt.title("Resized (Aspect Ratio Preserved)", fontsize=14)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pre_img)
        plt.title("Final Image Sent to TrOCR", fontsize=14)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("preprocessing_stages.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print("Preprocessing visualization saved as: preprocessing_stages.png")
    except Exception as e:
        print(f"Warning: Could not save preprocessing stages visualization: {e}")
else:
    print("Skipping preprocessing visualization (matplotlib not available)")

# -----------------------------
# RUN OCR
# -----------------------------
print("\n Running OCR recognition...")
try:
    text = recognize_image(pre_img)
    
    print("\n==============================")
    print("RECOGNIZED TEXT:")
    print("==============================")
    print(text)
    print("==============================")
except Exception as e:
    print(f"Error during OCR recognition: {e}")
    sys.exit(1)

# -----------------------------
# SHOW ORIGINAL IMAGE
# -----------------------------
if MATPLOTLIB_AVAILABLE:
    try:
        orig = Image.open(IMAGE_PATH)
        plt.figure(figsize=(8,6))
        plt.imshow(orig)
        plt.axis("off")
        plt.title("Original Input Image", fontsize=16)
        plt.savefig("original_image.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print("Original image saved as: original_image.png")
    except Exception as e:
        print(f"Warning: Could not save original image visualization: {e}")
else:
    print("Skipping original image visualization (matplotlib not available)")

# -----------------------------
# TEXT → SPEECH (ORIGINAL)
# -----------------------------
if text.strip():
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        speech_path = os.path.join(output_dir, "original_speech.mp3")
        
        tts = gTTS(text=text, lang="en")
        tts.save(speech_path)
        print(f"\n Speech saved: {speech_path}")
    except Exception as e:
        print(f"Warning: Could not generate speech: {e}")
else:
    print("No text detected to speak.")

# -----------------------------
# TRANSLATION + SPEECH
# -----------------------------
target_lang = "hi"   # Hindi

if text.strip():
    if translator is None:
        print("Translation skipped: Translator not available")
    else:
        try:
            print(f"\n Translating to {target_lang}...")
            translated = translator.translate(text, dest=target_lang).text
            
            print("\n==============================")
            print("TRANSLATED TEXT:")
            print("==============================")
            print(translated)
            print("==============================")
            
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            translated_speech_path = os.path.join(output_dir, "translated_speech.mp3")
            
            tts = gTTS(text=translated, lang=target_lang)
            tts.save(translated_speech_path)
            print(f"\n Translated speech saved: {translated_speech_path}")
        except Exception as e:
            print(f"Translation failed: {e}")
            print("Note: Translation may fail due to API rate limits. The OCR text is still available above.")
else:
    print("No text to translate.")

print("\n Processing complete!")
