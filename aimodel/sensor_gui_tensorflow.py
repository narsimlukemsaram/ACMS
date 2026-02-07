#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
Project      : Agricultural Environment Monitoring System
Hardware     : Raspberry Pi 5, Camera Module 3, BMP280, BH1750 Sensor
Dependencies : PyQt6, Picamera2, smbus2, numpy, tensorflow, opencv-python
Developed by : Suryameghan Lakkaraju
Date         : January 2026
-------------------------------------------------------------------------------
"""

import sys
import sensors_data as sd
from sensors_data import bus
import cv2
import numpy as np
import os
import yaml
import textwrap

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout, 
                             QSizePolicy, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

# TensorFlow imports
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Warning: tensorflow not found. AI features will be disabled.")
    load_model = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Update this path to your actual model file on the Pi
# Using absolute path resolution for robustness
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model_1_first.h5') 

# Classes sorted alphabetically as per flow_from_directory behavior
CLASS_LABELS = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border-radius: 8px;")
        self.setText("Camera Feed Loading...")

    def set_image(self, qt_img):
        """Scales the incoming QImage to fit the current label size."""
        lbl_w = self.width()
        lbl_h = self.height()
        if lbl_w > 0 and lbl_h > 0:
            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

class SensorCard(QGroupBox):
    def __init__(self, title, items):
        super().__init__(title)
        self.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout = QGridLayout()
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(15)
        
        self.value_labels = {}
        self.units = {}
        
        for i, (label_text, unit) in enumerate(items):
            # Label Name
            name_lbl = QLabel(label_text + ":")
            name_lbl.setFont(QFont("Segoe UI", 11))
            name_lbl.setStyleSheet("color: #333;")
            
            # Value Placeholder
            val_lbl = QLabel("--")
            val_lbl.setFont(QFont("Consolas", 12, QFont.Weight.Bold))

            val_lbl.setStyleSheet("color: #007acc; background: #eef; padding: 4px 8px; border-radius: 4px; border: 1px solid #dde; min-width: 100px;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # Unit
            unit_lbl = QLabel(unit)
            unit_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Light))
            unit_lbl.setStyleSheet("color: #000; border: none; font-weight: bold;")
            
            layout.addWidget(name_lbl, i, 0)
            layout.addWidget(val_lbl, i, 1)
            layout.addWidget(unit_lbl, i, 2)
            
            self.value_labels[label_text] = val_lbl
            
        self.setLayout(layout)
        
        self.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 1.2em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 14px;
            }
        """)

    def update_value(self, key, value):
        if key in self.value_labels:
            self.value_labels[key].setText(str(value))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agricultural Environment Monitoring (AI Enabled)")
        self.resize(1100, 700)
        
        # Load Remedies
        self.remedies = {}
        try:
            data_yaml_path = os.path.join(BASE_DIR, 'data.yaml')
            if os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.remedies = config.get('remedies', {})
        except Exception as e:
            print(f"Error loading remedies from data.yaml: {e}")

        self.static_image_mode = False
        self.current_frame = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout - Vertical to include Header
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Header ---
        header_lbl = QLabel("Autonomous Crop Monitoring System")
        header_lbl.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header_lbl.setStyleSheet("color: #2c3e50; padding-bottom: 10px;")
        header_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_lbl)

        # --- Main Card Container ---
        main_card = QWidget()
        main_card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
            }
        """)
        main_layout.addWidget(main_card)
        
        # --- Internal Content Layout ---
        content_layout = QHBoxLayout(main_card)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(30)

        # --- Left Col: Camera ---
        cam_container = QWidget()
        cam_layout = QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        
        cam_title = QLabel("Live AI Monitoring ")
        cam_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        cam_title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        
        self.video_display = VideoLabel()
        
        # Test Static Image Button
        self.btn_load_img = QPushButton("📁 Test Static Image")
        self.btn_load_img.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_load_img.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border-radius: 6px;
                padding: 10px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
        """)
        self.btn_load_img.clicked.connect(self.load_static_image)

        # Switch back to Camera Button
        self.btn_live_cam = QPushButton("🎥 Resume Live Camera")
        self.btn_live_cam.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_live_cam.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 6px;
                padding: 10px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #1e8449;
            }
        """)
        self.btn_live_cam.clicked.connect(self.resume_live_camera)
        self.btn_live_cam.hide() # Hidden by default
        
        cam_layout.addWidget(cam_title)
        cam_layout.addWidget(self.video_display, 1)
        cam_layout.addWidget(self.btn_load_img)
        cam_layout.addWidget(self.btn_live_cam)
        
        cam_container.setStyleSheet("border: none; background: transparent;")
        
        # --- Right Col: Sensors ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        # Environment Sensor Card
        env_items = [
            ("Temperature", "°C"),
            ("Humidity", "%RH"),
            ("Pressure", "Pa"),
            ("Light Intensity", "lux"),
            ("CO\u2082", "ppm")
        ]
        self.env_card = SensorCard("Environmental Data", env_items)
        
        # Soil Sensor Card
        soil_items = [
            ("pH Value", ""),
            ("Soil Moisture", "%")
        ]
        self.soil_card = SensorCard("Soil Status", soil_items)
        
        right_layout.addWidget(self.env_card)
        right_layout.addWidget(self.soil_card)
        right_layout.addStretch() 
        
        # Add Sensors to Left (Stretch 2)
        content_layout.addWidget(right_panel, stretch=2)
        
        # Add Camera to Right (Stretch 3)
        content_layout.addWidget(cam_container, stretch=3)

        # Initialize Model
        self.model = None
        self.load_ai_model()

        # Setup Camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.picam2 = None
        self.cap = self.setup_camera()
        self.timer.start(30) 

    def load_ai_model(self):
        """Loads the Keras model if available."""
        if load_model is None:
            return

        print(f"🔹 Attempting to load model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                print("✅ Model loaded successfully!")
                # Warmup
                dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
                self.model.predict(dummy, verbose=0)
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"⚠️ Model file not found at {MODEL_PATH}. Prediction disabled.")

    def setup_camera(self):
        print("Attempting camera initialization...")
        
        if Picamera2 is not None:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
                self.picam2.configure(config)
                self.picam2.start()
                print("Success: Camera opened using picamera2.")
                
                # Setup Sensors Data
                self.data_timer = QTimer()
                self.data_timer.timeout.connect(self.update_sensor_data)
                self.data_timer.start(2000)
                self.apply_theme()
                return "picamera2"
            except Exception as e:
                print(f"Picamera2 init failed (expected on non-Pi systems): {e}")
        
        # Fallback to OpenCV VideoCapture
        print("Falling back to cv2.VideoCapture(0)...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Success: Camera opened using cv2.VideoCapture(0).")
            # Setup Sensors Data (also needed here)
            self.data_timer = QTimer()
            self.data_timer.timeout.connect(self.update_sensor_data)
            self.data_timer.start(2000)
            self.apply_theme()
            return cap
        else:
            print("Error: Could not open any camera.")
            return None
        
    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f6f9;
            }
            QLabel {
                color: #2c3e50;
            }
        """)

    def load_static_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image for Detection", "", "Images (*.png *.jpg *.jpeg *.JPG)")
        if file_path:
            print(f"🔹 Processing Static Image: {file_path}")
            self.static_image_mode = True
            self.btn_live_cam.show()
            self.process_static_image(file_path)

    def resume_live_camera(self):
        self.static_image_mode = False
        self.btn_live_cam.hide()
        print("🎥 Resuming Live Camera Feed...")

    def process_static_image(self, img_path):
        if self.model is None:
            print("❌ Model not loaded. Cannot process image.")
            return

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            print(f"❌ Error: Could not read image at {img_path}")
            return

        # 1. Predict
        # Preprocess: Resize to 128x128
        img_input_res = cv2.resize(cv_img, (128, 128))
        img_input_res = cv2.cvtColor(img_input_res, cv2.COLOR_BGR2RGB)
        img_normalized = img_input_res.astype("float32") / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        preds = self.model.predict(img_input, verbose=0)
        idx = np.argmax(preds)
        label = CLASS_LABELS[idx]
        confidence = float(np.max(preds))
        remedy = self.remedies.get(label, "No specific remedy available.")

        # 2. Format output like demo_single_image.py
        h, w = cv_img.shape[:2]
        
        # Ensure minimum width for text readability in GUI
        display_w = max(w, 800)
        scale = display_w / w
        cv_img_resized = cv2.resize(cv_img, (0,0), fx=scale, fy=scale)
        h, w = cv_img_resized.shape[:2]

        # Draw Border
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(cv_img_resized, (0, 0), (w-1, h-1), color, 6)

        # Header
        header_bg_color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255) # Green or Orange
        text_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness_text = 2
        padding = 20
        line_spacing = 15
        
        header_text_lines = [
            f"Detected: {label.split('___')[-1]}",
            f"Confidence: {confidence*100:.2f}%"
        ]
        
        # Calculate wrapping
        approx_char_width = 22
        chars_per_line = max(25, int((w - 2*padding) / approx_char_width))
        wrapper = textwrap.TextWrapper(width=chars_per_line)
        wrapped_header_lines = []
        for line in header_text_lines:
            wrapped_header_lines.extend(wrapper.wrap(line))

        header_h = padding
        for line in wrapped_header_lines:
            (fw, fh), _ = cv2.getTextSize(line, font, font_scale, thickness_text)
            header_h += fh + line_spacing
        header_h += padding

        header_img = np.zeros((header_h, w, 3), dtype=np.uint8)
        header_img[:] = header_bg_color
        
        y = padding
        for line in wrapped_header_lines:
            (fw, fh), _ = cv2.getTextSize(line, font, font_scale, thickness_text)
            y += fh
            cv2.putText(header_img, line, (padding, y), font, font_scale, text_color, thickness_text, cv2.LINE_AA)
            y += line_spacing

        # Footer
        wrapper_footer = textwrap.TextWrapper(width=max(40, int(w / 15)))
        note_text = f"Action: {remedy}"
        footer_lines = wrapper_footer.wrap(note_text)
        footer_line_height = 35
        footer_h = (len(footer_lines) * footer_line_height) + (2 * padding)
        
        footer_img = np.zeros((footer_h, w, 3), dtype=np.uint8)
        footer_img[:] = (240, 240, 240)
        
        y = padding + 20
        for line in footer_lines:
            cv2.putText(footer_img, line, (padding, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            y += footer_line_height
            
        final_output = np.vstack((header_img, cv_img_resized, footer_img))
        # Final output is BGR, convert to RGB for Qt
        final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        
        # Update display
        h, w, ch = final_output_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(final_output_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_display.set_image(qt_img)

    def update_frame(self):
        if self.static_image_mode:
            return

        if self.cap is None:
            self.video_display.setText("No Camera Found")
            # For testing without camera, you might want to load a dummy image here
            return
        
        frame = None
        
        # Handle picamera2
        if self.cap == "picamera2" and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
            except Exception as e:
                print(f"Error capturing from picamera2: {e}")
                self.video_display.setText("Error: Camera capture failed")
                return

        # Handle cv2.VideoCapture
        elif isinstance(self.cap, cv2.VideoCapture):
             ret, frame = self.cap.read()
             if not ret:
                 return
             # OpenCV returns BGR, convert to RGB
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame is None:
            return
        
        # ---------------------------------------------------------
        # AI INFERENCE
        # ---------------------------------------------------------
        if self.model:
            try:
                # 1. Preprocess: Resize to 128x128
                img_small = cv2.resize(frame, (128, 128))
                
                # 2. Normalize [0, 1]
                img_normalized = img_small.astype("float32") / 255.0
                
                # 3. Add batch dimension (1, 128, 128, 3)
                img_input = np.expand_dims(img_normalized, axis=0)
                
                # 4. Predict
                preds = self.model.predict(img_input, verbose=0)
                
                # 5. Decode
                idx = np.argmax(preds)
                label = CLASS_LABELS[idx]
                confidence = float(np.max(preds))
                
                # Debug print to terminal
                
                # 6. Draw Overlay
                # Tightened threshold to 0.92 to avoid false positives on non-leaf subjects
                if confidence > 0.92: 
                    text_str = f"{label.split('___')[-1]} ({confidence*100:.1f}%)"
                    color = (0, 255, 0) # Green (RGB)
                else:
                    text_str = "Scanning for Crop..."
                    color = (255, 255, 0) # Yellow/Cyan-ish (RGB)
                
                # Draw text with outline for better visibility
                cv2.putText(frame, text_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3) # Black outline
                cv2.putText(frame, text_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)   # Color text
            
            except Exception as e:
                print(f"Inference Error: {e}")
        # ---------------------------------------------------------

        h, w = frame.shape[:2]
        ch = frame.shape[2] if len(frame.shape) == 3 else 1
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Smart Scaling
        lbl_w = self.video_display.width()
        lbl_h = self.video_display.height()
        
        if lbl_w > 0 and lbl_h > 0:
            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.video_display.setPixmap(scaled_pixmap)

    def update_sensor_data(self):
        t,p = sd.bmp280_get_data(bus)
        l = sd.bh1750_get_data(bus)
        
        # Simulate slight fluctuations
        import random
        h = 65.0 + random.uniform(-5, 5)
        co2 = 425 + random.uniform(-25, 25)
        ph = 6.8 + random.uniform(-0.2, 0.2)
        sm = 35.0 + random.uniform(-2, 2)

        self.env_card.update_value("Temperature", f"{t:8.2f}")
        self.env_card.update_value("Humidity", f"{h:8.2f}")
        self.env_card.update_value("Pressure", f"{p:8.2f}")
        self.env_card.update_value("Light Intensity", f"{l:>8.2f}")
        self.env_card.update_value("CO\u2082", f"{int(co2):4d}")
        
        self.soil_card.update_value("pH Value", f"{ph:.2f}")
        self.soil_card.update_value("Soil Moisture", f"{sm:.1f}")

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'data_timer'):
            self.data_timer.stop()
        if self.cap == "picamera2" and self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception as e:
                print(f"Error stopping picamera2: {e}")
        elif self.cap is not None and self.cap != "picamera2" and hasattr(self.cap, 'isOpened'):
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                print(f"Error releasing camera: {e}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
