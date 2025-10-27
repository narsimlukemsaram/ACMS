# ======================================================
# CNN Training Script — Enhanced (with Added Conv Layers)
# ======================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------------
# 1️⃣ Reproducibility
# ------------------------------------------------------
np.random.seed(1337)

# ------------------------------------------------------
# 2️⃣ Model Architecture (with extra Conv2D + BatchNorm layers)
# ------------------------------------------------------
classifier = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3 (added as improvement)
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 4 (optional lightweight layer for deeper feature extraction)
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),

    # Fully Connected Layers
    Dense(256, activation='relu'),
    Dropout(0.5),

    # Output Layer (10 classes)
    Dense(10, activation='softmax')
])

# ------------------------------------------------------
# 3️⃣ Compile Model
# ------------------------------------------------------
optimizer = Adam(learning_rate=1e-4)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

classifier.summary()

# ------------------------------------------------------
# 4️⃣ Data Generators
# ------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_dir = 'train'  # 👈 update with your train folder
val_dir = 'val'      # 👈 update with your validation folder

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

# ------------------------------------------------------
# 5️⃣ Callbacks (auto-tuning & early stop)
# ------------------------------------------------------
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stop, lr_reduce]

# ------------------------------------------------------
# 6️⃣ Train the Model
# ------------------------------------------------------
epochs = 40

history = classifier.fit(
    training_set,
    validation_data=validation_set,
    epochs=epochs,
    callbacks=callbacks
)

# ------------------------------------------------------
# 7️⃣ Plot Training & Validation Graphs
# ------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_graphs.png')
plt.show()

# ------------------------------------------------------
# 8️⃣ Final Stats
# ------------------------------------------------------
final_val_acc = val_acc[-1] * 100
final_val_loss = val_loss[-1]

print(f"\n✅ Final Validation Accuracy: {final_val_acc:.2f}%")
print(f"✅ Final Validation Loss: {final_val_loss:.4f}")
print("📊 Graph saved as 'training_validation_graphs.png'")
print("💾 Best model saved as 'best_model.h5'")

