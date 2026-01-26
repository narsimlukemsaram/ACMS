#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
Project      : Agricultural Environment Monitoring System
Hardware     : Raspberry Pi 5, Camera Module 3, BMP280, BH1750 Sensor
Dependencies : PyQt6, Picamera2, smbus2, numpy
Developed by : Suryameghan Lakkaraju
Date         : January 2026
-------------------------------------------------------------------------------
"""

import sys
import sensors_data as sd
from sensors_data import bus

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout, 
                             QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
import numpy as np
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

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

        self.setWindowTitle("Agricultural Environment Monitoring")
        self.resize(1024, 600)
        
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
        
        cam_title = QLabel("Live Monitoring ")
        cam_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        cam_title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        
        self.video_display = VideoLabel()
        
        cam_layout.addWidget(cam_title)
        cam_layout.addWidget(self.video_display, 1)
        
        cam_container.setStyleSheet("border: none; background: transparent;")
        
        # Add Camera to Right (Stretch 3)
        # We will add it AFTER sensors to match the user's sketch (Sensors Left, Camera Right)
        # content_layout.addWidget(cam_container, stretch=3) # We add this later

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

        # Setup Camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.picam2 = None
        self.cap = self.setup_camera()
        self.timer.start(30) 

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
        
    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f4f6f9;
            }
            QLabel {
                color: #2c3e50;
            }
        """)

    def update_frame(self):
        if self.cap is None:
            self.video_display.setText("No Camera Found")
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

        if frame is None:
            return
        
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
