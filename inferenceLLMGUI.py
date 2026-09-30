from PIL import ExifTags
import os
import sys
import cv2
import time
import statistics
import subprocess
import torch
import threading
import queue
import speech_recognition as sr
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QWidget, QHBoxLayout, QVBoxLayout
from ultralytics import YOLO
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# Global Configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_1_PATH = get_resource_path("yolo26m_access_track.mlpackage")
MODEL_2_PATH = get_resource_path("yolov8m.mlpackage")
LLM_MODEL_PATH = get_resource_path("models/smolvlm2-500m") # https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
ANNOUNCEMENT_COOLDOWN = 1.0  # seconds between end of last TTS and next LLM inference

btn_style = """
    QPushButton {
        background-color: rgba(12, 16, 24, 0.88);
        color: #9AA0A6;
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 4px;
        padding: 9px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: rgba(25, 35, 50, 0.95);
        color: #FFFFFF;
        border: 1px solid #00E676;
    }
    QPushButton:checked {
        background-color: rgba(0, 230, 118, 0.22);
        color: #00FF88;
        border: 2px solid #00E676;
        font-weight: bold;
    }
"""

label_style = """
    QLabel {
        background-color: rgba(15, 15, 15, 0.85);
        color: #EEEEEE;
        border: 1px solid #444444;
        padding: 10px 22px;
        font-size: 18px;
        font-weight: bold;
    }
"""

fps_label_style = """
    QLabel {
        background-color: rgba(15, 15, 15, 0.85);
        color: #EEEEEE;
        border: 1px solid #444444;
        padding: 10px 36px;
        font-size: 24px;
        font-weight: bold;
    }
"""

class VoiceListenerThread(QThread):
    command_recognized = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_speaking = False

    def run(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        with sr.Microphone(device_index=1) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            recognizer.energy_threshold = 150
            recognizer.dynamic_energy_threshold = False
            print("[Voice] Ready! Listening for voice commands...\n")

            while hasattr(self.parent(), "running") and self.parent().running and not self.isInterruptionRequested():
                try:
                    audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=4.0)

                    if self.is_speaking:
                        continue

                    text = recognizer.recognize_google(audio).lower().strip()
                    print(f"\nVoice heard: '{text}'\n")
                    if text:
                        self.command_recognized.emit(text)
                except sr.WaitTimeoutError:
                    pass  # Normal loop timeout, ignore silently
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Voice Error: {e}")


class CVWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VITURE Glasses Stream")


        self.llm_model = AutoModelForImageTextToText.from_pretrained(LLM_MODEL_PATH, torch_dtype=torch.float16, local_files_only=True).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(LLM_MODEL_PATH, local_files_only=True)
        self.llm_model.eval()
        print("SmolVLM model initialized successfully.")
        
        screens = QApplication.screens()

        if len(screens) > 1:
            external = screens[1]
            self.setGeometry(external.geometry())
        self.showFullScreen()

        self.llm_busy = False
        self.last_announced = 0.0
        self.muted = False

        self.model_1 = YOLO(MODEL_1_PATH, task="detect")
        self.model_2 = YOLO(MODEL_2_PATH, task="detect")
        
        self.active_model_name = "YOLOv8m"
        self.model = self.model_2

        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)
        self.setCentralWidget(self.video_label)

        # Top left: Stream and Audio Status
        self.top_left_container = QWidget(self.video_label)
        tl_layout = QHBoxLayout(self.top_left_container)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel(self.top_left_container)
        self.status_label.setStyleSheet(label_style)
        tl_layout.addWidget(self.status_label)

        # Top right: FPS Telemetry and Other Performance Metrics
        self.top_right_container = QWidget(self.video_label)
        tr_layout = QHBoxLayout(self.top_right_container)
        tr_layout.setContentsMargins(0, 0, 0, 0)
        self.fps_status_label = QLabel(self.top_right_container)
        self.fps_status_label.setStyleSheet(fps_label_style)
        self.fps_status_label.setMinimumWidth(380)
        self.fps_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tr_layout.addWidget(self.fps_status_label)

        # Top center: Model Selector Buttons
        self.top_center_container = QWidget(self.video_label)
        tc_layout = QHBoxLayout(self.top_center_container)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_layout.setSpacing(8)

        select_tag_style = """
            QLabel {
                background-color: transparent;
                color: #FFFFFF;
                border: none;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
                padding-right: 4px;
            }
        """

        self.select_model_label = QLabel("SELECT MODEL:", self.top_center_container)
        self.select_model_label.setStyleSheet(select_tag_style)

        self.btn_model1 = QPushButton("YOLO26m Access", self.top_center_container)
        self.btn_model2 = QPushButton("YOLOv8m", self.top_center_container)

        self.btn_model1.setStyleSheet(btn_style)
        self.btn_model2.setStyleSheet(btn_style)
        self.btn_model1.setCheckable(True)
        self.btn_model2.setCheckable(True)
        self.btn_model2.setChecked(True)
        self.btn_model1.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_model2.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_model1.clicked.connect(self.select_model_1)
        self.btn_model2.clicked.connect(self.select_model_2)

        # Global shortcuts so ESC and Q work regardless of focused widget
        QShortcut(QKeySequence("Esc"), self, self.close)
        QShortcut(QKeySequence("Q"), self, self.close)

        tc_layout.addWidget(self.select_model_label)
        tc_layout.addWidget(self.btn_model1)
        tc_layout.addWidget(self.btn_model2)

        # Bottom right: LLM Status & Active Model Display
        self.bottom_right_container = QWidget(self.video_label)
        br_layout = QVBoxLayout(self.bottom_right_container)
        br_layout.setContentsMargins(0, 0, 0, 0)

        self.model_status_label = QLabel(self.bottom_right_container)
        self.model_status_label.setStyleSheet(label_style)
        br_layout.addWidget(self.model_status_label, 0, alignment=Qt.AlignmentFlag.AlignRight)

        # Hide overlay containers initially until positions are calculated on first frame
        self.top_left_container.hide()
        self.top_right_container.hide()
        self.top_center_container.hide()
        self.bottom_right_container.hide()

        self.latest_description = "Waiting for scene description..."

        self.detections = []

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.latest_raw_frame = None
        self.running = True
        threading.Thread(target=self._webcam_reader, daemon=True).start()

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out = self.create_output_writer(frame_width, frame_height)
        self.video_queue = queue.Queue(maxsize=30)
        threading.Thread(target=self._video_writer, daemon=True).start()

        self.voice_thread = VoiceListenerThread(self)
        self.voice_thread.command_recognized.connect(self.handle_voice_command)
        self.voice_thread.start()

        self.last_frame_time = time.time()
        self.fps = 0.0
        self.fps_history = []
        self.mean_fps = 0.0
        self.std_fps = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(30)
        self.timer.start()

    def _webcam_reader(self):
        while self.running:
            try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.latest_raw_frame = frame
            except Exception as e:
                time.sleep(0.01)
    
    def create_output_writer(self, width, height):
        os.makedirs("webcams", exist_ok=True)

        timestamp = str(int(time.time()))
        self.output_filename = f"webcams/webcam_{timestamp}.avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            self.output_filename,
            fourcc,
            30,
            (width, height)
        )
        return out

    def _video_writer(self):
        while self.running:
            try:
                frame = self.video_queue.get(timeout=0.1)
                self.out.write(frame)
                self.video_queue.task_done()
            except queue.Empty:
                continue

    def handle_voice_command(self, command_text: str):
        if "mute" in command_text or "quiet" in command_text:
            self.muted = True
            subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        elif "unmute" in command_text or "sound on" in command_text:
            self.muted = False
        elif "model one" in command_text or "access" in command_text:
            self.select_model_1()
            self.speak("Switched to access model.")
        elif "model two" in command_text or "yolo" in command_text:
            self.select_model_2()
            self.speak("Switched to yolo model.")
        elif "close" in command_text or "quit" in command_text or "exit" in command_text:
            self.close()

    def speak(self, text):
        if self.muted:
            return
        
        # Mute/pause voice command listening while TTS is speaking
        if self.voice_thread is not None:
            self.voice_thread.is_speaking = True

        subprocess.run(["say", text])

        # Short pause to let room reverb / speaker echo settle before listening again
        time.sleep(0.3)
        if self.voice_thread is not None:
            self.voice_thread.is_speaking = False

    def run_yolo_model_inference(self, frame):
        self.detections = [] # Reset detections
        h, w = frame.shape[:2]

        results = self.model.predict(frame, imgsz=640, conf=0.5, verbose=False, device="cpu")

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
                class_name = self.model.names[class_id]
                self.detections.append((class_name, score.item()))
                label = f"{class_name}: {score.item()*100:.1f}%"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )
        return frame

    def update_frame(self):
        t_start = time.time()

        # Camera Read
        t0 = time.time()
        if self.latest_raw_frame is None:
            return
        frame = self.latest_raw_frame.copy()
        t_read = (time.time() - t0) * 1000

        # YOLO Inference & Bounding Boxes
        t0 = time.time()
        frame = self.run_yolo_model_inference(frame)
        t_yolo = (time.time() - t0) * 1000

        current_time = time.time()
        if (current_time - self.last_announced) > ANNOUNCEMENT_COOLDOWN:
            if not self.llm_busy:
                self.llm_busy = True
                detections_copy = list(self.detections)
                raw_frame = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else frame.copy()
                pil_image = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
                threading.Thread(
                    target=self.run_llm_inference, 
                    args=(pil_image, detections_copy),
                    daemon=True
                ).start()

        # Step 4: Overlays
        t0 = time.time()
        frame = self.draw_subtitle_banner(frame)

        # Update text labels
        audio_label = "MUTED" if self.muted else "AUDIO ON"
        self.status_label.setText(f"🔴 REC   |   {audio_label}")
        self.fps_status_label.setText(f"FPS: {self.fps:.1f} (Avg: {self.mean_fps:.1f})")
        self.model_status_label.setText(f"VLM: SmolVLM2-500M   |   YOLO MODEL:  {self.active_model_name}")

        if not self.top_left_container.isVisible():
            self.reposition_overlays()
            self.top_left_container.show()
            self.top_right_container.show()
            self.top_center_container.show()
            self.bottom_right_container.show()

        t_overlays = (time.time() - t0) * 1000

        # Step 5: PyQt Image Conversion & GPU Render
        t0 = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))
        t_qt_render = (time.time() - t0) * 1000

        if not self.video_queue.full():
            self.video_queue.put(frame)

        # Calculate Total Frame Time, Instantaneous FPS, Mean FPS & Std Dev FPS
        t_total = (time.time() - t_start) * 1000
        elapsed_since_last_frame = time.time() - self.last_frame_time
        self.last_frame_time = time.time()
        self.fps = 1.0 / elapsed_since_last_frame if elapsed_since_last_frame > 0 else 0

        if self.fps > 0:
            self.fps_history.append(self.fps)
            stats_data = self.fps_history[5:] if len(self.fps_history) > 10 else self.fps_history
            self.mean_fps = statistics.mean(stats_data)
            self.std_fps = statistics.stdev(stats_data) if len(stats_data) > 1 else 0.0

        print(
            f"Read: {t_read:4.1f} ms | "
            f"YOLO: {t_yolo:4.1f} ms | "
            f"Overlays: {t_overlays:3.1f} ms | "
            f"Qt Render: {t_qt_render:4.1f} ms | "
            f"Total: {t_total:4.1f} ms | "
            f"FPS: {self.fps:4.1f} | "
            f"Mean FPS: {self.mean_fps:4.1f} ± {self.std_fps:4.2f}"
        )

    def run_llm_inference(self, pil_image, detections):
        print("\n\nrunning llm...\n\n")
        if self.llm_model is None or self.processor is None:
            print("LLM Model is not initialized.")
            return
        try:
            if not detections:
                print("Detections not found")
                self.latest_description = "No relevant objects were detected in the scene."
                return

            det_str = ", ".join([f"{cls}" for cls, _ in detections])

            prompt_text = (
                f"The following objects were detected in the camera view: {det_str}. Using visual context from the full image, describe the scene in one concise natural sentence. Only mention the detected objects and their context. If no objects were detected, just say 'No relevant objects detected in the scene'."
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(DEVICE, dtype=torch.float16)

            with torch.inference_mode():
                generated_ids = self.llm_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=40
                )

            num_prompt_tokens = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, num_prompt_tokens:]
            description = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            
            print(description, end="\n")
            self.latest_description = description
            print(f"\nDescription: {len(detections)} {self.latest_description}\n")

            self.speak(description)
            self.last_announced = time.time()
        except Exception as e:
            print(f"\nError: {e}\n")
        finally:
            self.llm_busy = False
    
    def draw_subtitle_banner(self, frame):
        if not self.latest_description:
            return frame
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        line_spacing = 30
        margin_x = 20
        max_text_width = w - (margin_x * 2)
        
        text = f"Description: {self.latest_description}"

        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            (test_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if test_w <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        banner_height = max(80, 40 + len(lines) * line_spacing)
        y1, y2 = h - banner_height, h

        roi = frame[y1:y2, 0:w]
        banner = roi.copy()
        cv2.rectangle(banner, (0, 0), (w, y2 - y1), (15, 15, 20), -1)

        alpha = 0.5
        cv2.addWeighted(banner, alpha, roi, 1 - alpha, 0, roi)

        text_y = y1 + 35
        for line in lines:
            cv2.putText(frame, line, (margin_x, text_y), font, font_scale, color, thickness)
            text_y += line_spacing
        return frame

    def keyPressEvent(self, event):
        key = event.key()

        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif key == Qt.Key.Key_M:
            self.muted = not self.muted
            if self.muted:
                subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'top_left_container'):
            self.reposition_overlays()

    def reposition_overlays(self):
        # Position Top-Left Corner
        self.top_left_container.adjustSize()
        self.top_left_container.move(0, 300)

        # Position Top-Right Corner
        self.top_right_container.adjustSize()
        tr_x = self.video_label.width() - self.top_right_container.width() - 15
        self.top_right_container.move(max(0, tr_x), 300)

        # Position Top-Center
        self.top_center_container.adjustSize()
        tc_x = (self.video_label.width() - self.top_center_container.width()) // 2
        self.top_center_container.move(max(0, tc_x), 300)

        # Position Bottom-Right Corner
        self.bottom_right_container.adjustSize()
        br_x = self.video_label.width() - self.bottom_right_container.width()
        br_y = self.video_label.height() - self.bottom_right_container.height() - 150
        self.bottom_right_container.move(max(0, br_x), max(0, br_y))
    
    def select_model_1(self):
        self.model = self.model_1
        self.active_model_name = "YOLO26m Access"
        self.fps_history.clear()
        self.btn_model1.setChecked(True)
        self.btn_model2.setChecked(False)

    def select_model_2(self):
        self.model = self.model_2
        self.active_model_name = "YOLOv8m"
        self.fps_history.clear()
        self.btn_model1.setChecked(False)
        self.btn_model2.setChecked(True)

    def closeEvent(self, event):
        self.running = False
        if hasattr(self, 'voice_thread') and self.voice_thread.isRunning():
            self.voice_thread.requestInterruption()
            self.voice_thread.quit()
            if not self.voice_thread.wait(1000):
                self.voice_thread.terminate()
                self.voice_thread.wait()
        time.sleep(0.05)
        self.cap.release()
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        self.out.release()
        self.timer.stop()
        print(f"Output saved to {self.output_filename}")

        # Print Comprehensive Session FPS Performance Summary
        if self.fps_history:
            stats_data = self.fps_history[5:] if len(self.fps_history) > 10 else self.fps_history
            mean_f = statistics.mean(stats_data)
            std_f = statistics.stdev(stats_data) if len(stats_data) > 1 else 0.0
            median_f = statistics.median(stats_data)
            min_f = min(stats_data)
            max_f = max(stats_data)
            print("\n" + "=" * 55)
            print(f"       FPS PERFORMANCE SUMMARY ({self.active_model_name})       ")
            print("=" * 55)
            print(f"Total Frames Processed: {len(self.fps_history)}")
            print(f"Mean FPS:               {mean_f:.2f} FPS")
            print(f"Median FPS:             {median_f:.2f} FPS")
            print(f"Standard Deviation (σ): {std_f:.2f} FPS")
            print(f"Min / Max FPS:          {min_f:.1f} / {max_f:.1f} FPS")
            print("=" * 55 + "\n")

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CVWindow()
    window.show()
    sys.exit(app.exec())