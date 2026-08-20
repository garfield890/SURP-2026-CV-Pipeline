# SURP 2026 - Computer Vision Pipeline

A real-time computer vision and multimodal perception pipeline featuring YOLO object detection, local Qwen2.5 LLM, Text-to-Speech narration, and a PyQt6 interactive GUI.

---

## Features

- **Real-Time Object Detection**: High-performance detection using YOLOv8 / YOLO26 models with Apple Silicon GPU (`mps`) and Apple Neural Engine (`CoreML .mlpackage`) acceleration.
- **Multimodal & LLM Scene Understanding**: Summarizes detected objects and visual context into natural language descriptions using Qwen2.5-0.5B.
- **Interactive PyQt6 GUI**: Live camera stream, detection toggles, dual-model switching, hotkeys, and voice command recognition.
- **Audio Feedback & TTS**: Spoken scene announcements and voice feedback.
- **CoreML Model Exporting**: Scripts to convert PyTorch checkpoints into `.mlpackage` for optimized on-device inference.

---

## Project Structure

```text
├── inferenceLLMGUI.py     # Main PyQt6 GUI application with YOLO + LLM + Voice
├── inferenceLLM.py        # CLI camera stream with YOLO + Cloud LLM scene narration
├── inferenceNoLLM.py      # Lightweight YOLO-only detection with audio announcements
├── models.py              # Script to download and cache local Qwen2.5 LLM weights
├── export_model.py        # Converts YOLO PyTorch models to CoreML (.mlpackage)
├── test.py                # Standalone test script for LLM inference on detections
├── requirements.txt       # Python dependencies
└── .gitignore             # Git ignore rules for virtualenvs, models, and caches
```

---

## Getting Started

### 1. Prerequisites & Virtual Environment

Python 3.10+ is recommended. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: On macOS, ensure you have system microphone permissions enabled if using speech recognition.)*

### 3. Model Setup

1. **Download Local LLM (Qwen2.5):**
   ```bash
   python models.py
   ```
2. **(Optional) Export YOLO to CoreML:**
   Place your `yolov8m.pt` or `yolo26m_access_track.pt` weights in the project root, then run:
   ```bash
   python export_model.py
   ```

---

## Usage

### Run the PyQt6 GUI Application
```bash
python inferenceLLMGUI.py
```

### Run Command-Line Inference
- **With LLM Scene Narration:**
  ```bash
  python inferenceLLM.py
  ```
- **Object Detection Only:**
  ```bash
  python inferenceNoLLM.py
  ```