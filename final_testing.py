
import os
import cv2
import numpy as np
import yaml
import textwrap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuration
MODEL_PATH = '/home/kotesh/Downloads/archive/tomato/best_model_1_first.h5'
TEST_DIR = 'test'
DATA_YAML = 'data.yaml'
OUTPUT_DIR = 'final_testing_results'
IMG_SIZE = (128, 128)

# Load Configuration
print(f"Loading configuration from {DATA_YAML}...")
with open(DATA_YAML, 'r') as f:
    config = yaml.safe_load(f)

# Use the names from yaml, ensuring they match the training order (usually alphabetical)
class_names = sorted(config['names'])
remedies = config.get('remedies', {})

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Create Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process Images
print("Processing images from test folder...")
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

count = 0
for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(image_extensions):
            img_path = os.path.join(root, file)
            
            try:
                # 1. Prediction Step
                # Load image for prediction (using keras helper for correct resizing/processing)
                pil_img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(pil_img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Normalize
                
                # Predict
                predictions = model.predict(img_array, verbose=0)
                class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                predicted_class = class_names[class_idx]
                remedy = remedies.get(predicted_class, "No specific remedy available.")
                
                # 2. Visualization Step
                # Load using OpenCV for drawing
                cv_img = cv2.imread(img_path)
                if cv_img is None: continue
                
                h, w = cv_img.shape[:2]
                
                # Draw Box (Green border around image)
                color = (0, 255, 0) # Green
                thickness = 2
                cv2.rectangle(cv_img, (0, 0), (w-1, h-1), color, thickness)
                
                # Draw Text (Class + Confidence) at top
                label_text = f"{predicted_class} ({confidence*100:.1f}%)"
                # Add background for text readability
                (fw, fh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(cv_img, (0, 0), (fw + 10, fh + 15), color, -1) # Filled rectangle
                cv2.putText(cv_img, label_text, (5, fh + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # 3. Add Footer with Remedy Note
                footer_height = 80
                # Scale footer height if image is very large or text is long, but fixed is fine for now
                footer = np.zeros((footer_height, w, 3), dtype=np.uint8)
                footer[:] = (240, 240, 240) # Light gray background
                
                # Wrap text to fit width
                wrapper = textwrap.TextWrapper(width=int(w / 8)) # Approx char width logic
                wrapper.placeholder = "..."
                
                note_text = f"Action: {remedy}"
                wrapped_lines = wrapper.wrap(note_text)
                
                y_offset = 20
                font_scale = 0.4
                if w > 300: font_scale = 0.5 
                
                for line in wrapped_lines:
                    cv2.putText(footer, line, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
                    y_offset += 20
                    if y_offset > footer_height: break
                
                # Stack image and footer
                final_output = np.vstack((cv_img, footer))
                
                # Save
                # Helper to keep unique names if multiple files have same name in different subdirs
                base_name = os.path.basename(file)
                unique_name = f"{count}_{base_name}" 
                save_path = os.path.join(OUTPUT_DIR, unique_name)
                cv2.imwrite(save_path, final_output)
                
                count += 1
                if count % 20 == 0:
                    print(f"Processed {count} images...")

            except Exception as e:
                print(f"Failed to process {file}: {e}")

print(f"Processing complete. {count} images saved to '{os.path.abspath(OUTPUT_DIR)}'.")
