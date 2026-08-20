# -*- coding: utf-8 -*-
"""
Updated on July 30, 2025
Author: Fahim
"""

import os
import sys
import cv2
from ultralytics import YOLO
import time
import subprocess
import threading
import torch
import moondream as md
from PIL import Image

def get_api_key():
    api_key = os.environ.get("MOONDREAM_API_KEY")
    if api_key:
        return api_key
    
    if hasattr(sys, '_MEIPASS'):
        key_path = os.path.join(sys._MEIPASS, "secret_key.dat")
    else:
        key_path = "secret_key.dat"

    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()
    return ""

# Global Configuration
WINDOW_NAME = "Preview Window"
MODEL_PATH = "yolo26m_access_track.pt"
ANNOUNCEMENT_COOLDOWN = 5.0  # seconds between audio announcements
MOONDREAM_API_KEY = get_api_key()

# State variables
speech_thread = None
llm_busy = False
last_announced = 0.0

# ----------------------------
# Text-to-Speech & LLM Helper Functions
# ----------------------------
def speak(text):
    subprocess.run(["say", text])

# Get Moondream model from cloud and call API through Python SDK
try:
    llm_model = md.vl(api_key=MOONDREAM_API_KEY)
    print("Moondream Cloud API client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Moondream client: {e}")

def run_llm_inference(pil_image):
    global llm_busy, speech_thread
    try:
        description = llm_model.query(
            pil_image,
            "Describe the scene in exactly one concise sentence."
        )["answer"]
        print(f"\nDescription: {description}\n")
        speech_thread = threading.Thread(target=speak, args=(description,), daemon=True)
        speech_thread.start()
    except Exception as e:
        print(f"\nError: {e}\n")
    finally:
        llm_busy = False

# ----------------------------
# Load YOLO model on GPU
# ----------------------------
model = YOLO(MODEL_PATH)
model.to("mps")

# ----------------------------
# Open webcam
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Optional: Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Read one frame to determine image size
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture frame from webcam.")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]

# Display initial dummy frame and move to external monitor
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.imshow(WINDOW_NAME, frame)
cv2.moveWindow(WINDOW_NAME, 1920, -120)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.waitKey(1)

# ----------------------------
# Prepare output video
# ----------------------------

# Most webcams don't report FPS reliably
fps = 30

timestamp = str(int(time.time()))
output_filename = f"webcams/webcam_{timestamp}.avi"

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    output_filename,
    fourcc,
    fps,
    (frame_width, frame_height)
)

print("Press ESC to quit.")

# Webcam thread
latest_frame = None

def webcam_reader():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame

threading.Thread(target=webcam_reader, daemon=True).start()

frame_count = 0
# ----------------------------
# Main loop
# ----------------------------
while True:
    if latest_frame is None:
        continue
    frame = latest_frame.copy()

    frame_count += 1
    if frame_count % 60 == 0:
        torch.mps.empty_cache()

    start_time = time.time()

    img_boxes = frame.copy()
    h, w = frame.shape[:2]

    results = model.predict(img_boxes, conf=0.5, verbose=False)

    for result in results:
        for score, cls, bbox in zip(
                result.boxes.conf,
                result.boxes.cls,
                result.boxes.xyxy):

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = map(int, [x1.item(), y1.item(), x2.item(), y2.item()])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            class_id = int(cls.item())
            class_name = model.names[class_id]
            label = f"{class_name}: {score.item()*100:.1f}%"

            cv2.rectangle(
                img_boxes,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                img_boxes,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

    current_time = time.time()
    if (current_time - last_announced) > ANNOUNCEMENT_COOLDOWN:
        if not llm_busy:
            # Check if the previous speech is still playing to avoid overlapping voices
            if speech_thread is None or not speech_thread.is_alive():
                llm_busy = True
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                llm_thread = threading.Thread(target=run_llm_inference, args=(pil_image,), daemon=True)
                llm_thread.start()
                last_announced = current_time

    out.write(img_boxes)

    elapsed_time = time.time() - start_time
    fps_measured = 1.0 / elapsed_time if elapsed_time > 0 else 0
    print(f"Inference & Drawing: {elapsed_time*1000:.1f} ms | FPS: {fps_measured:.1f}")

    cv2.imshow(WINDOW_NAME, img_boxes)

    # Exit on ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved to {output_filename}")