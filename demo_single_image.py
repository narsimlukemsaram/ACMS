
import os
import cv2
import numpy as np
import yaml
import textwrap
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuration
MODEL_PATH = 'models/best_model_1_first.h5'
DATA_YAML = 'data.yaml'
IMG_SIZE = (128, 128)
DEFAULT_TEST_IMAGE = 'test/Tomato___Tomato_Yellow_Leaf_Curl_Virus/d8f24829-ea0b-4d97-a809-687a781d684c___UF.GRC_YLCV_Lab 09478.JPG' # Example

def main(img_path):
    print(f"🔹 Processing Image: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"❌ Error: Image file not found at {img_path}")
        return

    # 1. Load Data/Model
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = sorted(config['names'])
    remedies = config.get('remedies', {})

    print(f"🔹 Loading Model...")
    model = load_model(MODEL_PATH)
    
    # 2. Predict
    pil_img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = class_names[class_idx]
    remedy = remedies.get(predicted_class, "No specific remedy available.")
    
    print(f"✅ Prediction: {predicted_class} ({confidence:.2%})")

    # 3. Visualization
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print(f"❌ Error: Could not read image at {img_path}")
        return

    h, w = cv_img.shape[:2]
    
    # Upscale logic (Ensure min width for text readability)
    target_min_width = 600
    if w < target_min_width:
        scale = target_min_width / w
        cv_img = cv2.resize(cv_img, (0,0), fx=scale, fy=scale)
        h, w = cv_img.shape[:2]

    # Draw Border on the image itself
    color = (0, 255, 0) # Green
    thickness = 4
    cv2.rectangle(cv_img, (0, 0), (w-1, h-1), color, thickness)

    # --- Header Generation ---
    # Stack header ON TOP of image to avoid obscuring content or cutting off text
    header_bg_color = (0, 255, 0) # Green background
    text_color = (0, 0, 0) # Black text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness_text = 2
    padding = 15
    line_spacing = 10
    
    header_text_lines = [
        f"Detected: {predicted_class}",
        f"Confidence: {confidence*100:.2f}%"
    ]
    
    # Calculate required header text wrapping
    approx_char_width = 18 # Estimation for font_scale 0.8
    chars_per_line = max(20, int((w - 2*padding) / approx_char_width))
    wrapper = textwrap.TextWrapper(width=chars_per_line)
    
    wrapped_header_lines = []
    for line in header_text_lines:
        wrapped_header_lines.extend(wrapper.wrap(line))

    # Calculate precise header height
    header_h = padding
    for line in wrapped_header_lines:
        (fw, fh), _ = cv2.getTextSize(line, font, font_scale, thickness_text)
        header_h += fh + line_spacing
    header_h += padding # Bottom padding
    
    # Create Header Image
    header_img = np.zeros((header_h, w, 3), dtype=np.uint8)
    header_img[:] = header_bg_color
    
    # Draw text on header
    y = padding
    for line in wrapped_header_lines:
        (fw, fh), _ = cv2.getTextSize(line, font, font_scale, thickness_text)
        y += fh
        cv2.putText(header_img, line, (padding, y), font, font_scale, text_color, thickness_text, cv2.LINE_AA)
        y += line_spacing

    # --- Footer Generation ---
    wrapper_footer = textwrap.TextWrapper(width=int(w / 12)) # Roughly w/12 for smaller font
    note_text = f"Action: {remedy}"
    footer_lines = wrapper_footer.wrap(note_text)
    
    footer_line_height = 30
    footer_h = (len(footer_lines) * footer_line_height) + (2 * padding)
    
    footer_img = np.zeros((footer_h, w, 3), dtype=np.uint8)
    footer_img[:] = (240, 240, 240) # Light gray
    
    y = padding + 15
    for line in footer_lines:
        cv2.putText(footer_img, line, (padding, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += footer_line_height
        
    # Stack All (Header + Image + Footer)
    final_output = np.vstack((header_img, cv_img, footer_img))
    
    save_path = "demo_result.jpg"
    cv2.imwrite(save_path, final_output)
    print(f"💾 Result saved to: {os.path.abspath(save_path)}")
    
    # Try to show image (will work if run locally by user with display)
    try:
        from PIL import Image
        img_show = Image.open(save_path)
        img_show.show()
        print("🖼️ Opening image viewer...")
    except Exception as e:
        print("Could not open image viewer automatically.")

if __name__ == "__main__":
    # Allow user to pass image path as argument, else use default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Use a hardcoded example from existing test files if not provided
        img_path = DEFAULT_TEST_IMAGE
        
    main(img_path)
