# ============================================
# CNN Test Script — with Bounding Boxes & CSV
# ============================================

import os
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------
# 1. Config
# --------------------------------------------
model_path = '/home/kotesh/Downloads/archive/tomato/results/version_1/best_model_1_first.h5'
test_dir = 'test'  # 👈 update this path
batch_size = 32
output_csv = 'test_predictions_with_boxes.csv'

# Output directories
classified_dir = 'classified_images_2'
misclassified_dir = 'misclassified_images_2'
os.makedirs(classified_dir, exist_ok=True)
os.makedirs(misclassified_dir, exist_ok=True)

# --------------------------------------------
# 2. Load Model
# --------------------------------------------
print(f"🔹 Loading model from: {model_path}")
model = load_model(model_path)

# --------------------------------------------
# 3. Test Data Generator
# --------------------------------------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())
file_paths = test_generator.filepaths

# --------------------------------------------
# 4. Predict
# --------------------------------------------
print("🔹 Predicting on test dataset...")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# --------------------------------------------
# 5. Accuracy
# --------------------------------------------
test_accuracy = accuracy_score(true_classes, predicted_classes)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------
# 6. CSV & Image Saving
# --------------------------------------------
print(f"💾 Saving results and annotated images...")

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'image_path',
        'actual_class',
        'predicted_class',
        'confidence',
        'flag',
        'bbox_xmin',
        'bbox_ymin',
        'bbox_xmax',
        'bbox_ymax'
    ])

    for idx, (path, true_idx, pred_idx) in enumerate(zip(file_paths, true_classes, predicted_classes)):
        actual_class = class_labels[true_idx]
        predicted_class = class_labels[pred_idx]
        confidence = float(np.max(predictions[idx]))
        flag = (actual_class == predicted_class)

        # Bounding box: entire image (classification models don’t predict boxes)
        xmin, ymin, xmax, ymax = 0, 0, 127, 127

        # Read and draw on image
        image = cv2.imread(path)
        if image is not None:
            h, w = image.shape[:2]
            # Adjust bbox to original image size
            xmin, ymin, xmax, ymax = 0, 0, w - 1, h - 1
            color = (0, 255, 0) if flag else (0, 0, 255)
            label_text = f"{predicted_class} ({confidence*100:.1f}%)"

            # Draw rectangle and text
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Save to appropriate folder
            dest_folder = classified_dir if flag else misclassified_dir
            class_subdir = os.path.join(dest_folder, f"{actual_class}_as_{predicted_class}")
            os.makedirs(class_subdir, exist_ok=True)
            dest_path = os.path.join(class_subdir, os.path.basename(path))
            cv2.imwrite(dest_path, image)

        # Write to CSV
        writer.writerow([
            path,
            actual_class,
            predicted_class,
            confidence,
            flag,
            xmin, ymin, xmax, ymax
        ])

print("✅ CSV and annotated images created successfully!")

# --------------------------------------------
# 7. Confusion Matrix & Accuracy Plots
# --------------------------------------------
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(10, 10))
disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
plt.title(f'Confusion Matrix (Accuracy: {test_accuracy * 100:.2f}%)')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png')
plt.show()

# Per-class accuracy
correct_flags = [a == p for a, p in zip(true_classes, predicted_classes)]
accuracy_per_class = []
for i, label in enumerate(class_labels):
    indices = np.where(true_classes == i)[0]
    class_acc = np.mean([correct_flags[j] for j in indices]) if len(indices) > 0 else 0
    accuracy_per_class.append(class_acc * 100)

plt.figure(figsize=(10, 5))
plt.bar(class_labels, accuracy_per_class, color='skyblue')
plt.title('Per-Class Accuracy (%)')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_classwise_accuracy.png')
plt.show()

print("📊 Graphs saved: test_confusion_matrix.png & test_classwise_accuracy.png")

