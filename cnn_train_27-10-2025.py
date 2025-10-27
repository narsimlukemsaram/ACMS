# =====================================
# Enhanced CNN Training Script
# =====================================

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
np.random.seed(1337)

# =====================================
# 1. Define Model Architecture
#  Added BatchNormalization, dense layer to 256 neurons
# =====================================
classifier = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

# =====================================
# 2. Compile Model
# =====================================
#optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)  ### updated with AdamW optimizer from normal adam optimizer
optimizer = Adam(learning_rate=1e-3)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# =====================================
# 3. Data Preprocessing and Augmentation
# =====================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Point to your data directories
train_dir = 'train'
val_dir = 'val'

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_set = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# =====================================
# 4. Callbacks for Efficient Training added improvement here
# =====================================
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# =====================================
# 5. Train the Model
# =====================================
history = classifier.fit(
    training_set,
    epochs=50,                   # EarlyStopping will halt earlier if needed
    validation_data=validation_set,
    callbacks=[checkpoint, early_stop, reduce_lr] ### added more callbacks 
)

# =====================================
# 6. Evaluate and Visualize Results helps in model overfitting and stabilizing
# =====================================
# Plot training history
plt.figure(figsize=(10, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# =====================================
# 7. Evaluate Best Model
# =====================================
val_loss, val_acc = classifier.evaluate(validation_set)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"✅ Final Validation Loss: {val_loss:.4f}")

