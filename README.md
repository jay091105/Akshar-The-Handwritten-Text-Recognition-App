# Handwritten OCR Web Application with FastAPI

A web application for recognizing handwritten text from images, translating it to multiple languages, and converting it to speech using FastAPI.

## Features

- **OCR Recognition**: 
  - EasyOCR for text detection and recognition
  - TrOCR (Transformer-based OCR) for handwritten text
  - Combined approach for better accuracy

- **Translation**: Translate recognized text to multiple languages (Hindi, Spanish, French, German, Chinese, Japanese, Arabic)

- **Text-to-Speech**: Convert translated text to speech using Google TTS

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Note**: For TrOCR, the models will be downloaded automatically on first run from Hugging Face.

3. **For EasyOCR**: The model files will be downloaded automatically on first run.

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the web interface:**
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## API Endpoints

### 1. `GET /`
   - Serves the HTML frontend interface

### 2. `POST /recognize`
   - Upload an image and recognize text
   - **Parameters:**
     - `file`: Image file (multipart/form-data)
     - `method`: Recognition method - "easyocr", "trocr", or "combined"
   - **Returns:** JSON with recognized text

### 3. `POST /translate`
   - Translate text and generate speech
   - **Parameters:**
     - `text`: Text to translate
     - `target_lang`: Target language code (e.g., "hi", "es", "fr")
   - **Returns:** JSON with translated text and audio URL

### 4. `GET /audio/{filename}`
   - Serve generated audio files

### 5. `GET /health`
   - Health check endpoint

## Usage

1. Open the web interface at `http://localhost:8000`
2. Upload an image containing handwritten text
3. Select recognition method (EasyOCR, TrOCR, or Combined)
4. Click "Recognize Text" to extract text from the image
5. Select target language and click "Translate & Generate Speech"
6. Listen to the translated text

## Supported Languages for Translation

- English (en)
- Hindi (hi)
- Spanish (es)
- French (fr)
- German (de)
- Chinese Simplified (zh-cn)
- Japanese (ja)
- Arabic (ar)

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- At least 4GB RAM (8GB+ recommended)

## Notes

- First run will download model files (TrOCR ~600MB, EasyOCR models)
- Processing time depends on image size and complexity
- GPU acceleration is automatically used if available

## Troubleshooting

- If translation fails, it might be due to googletrans being unofficial. Consider using Google Cloud Translation API for production.
- If models fail to load, ensure you have sufficient disk space and internet connection for model downloads.

