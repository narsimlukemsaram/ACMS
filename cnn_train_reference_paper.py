from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
import numpy as np
from keras.callbacks import ModelCheckpoint

# Initialize the CNN
np.random.seed(1337)
classifier = Sequential()

# First block
classifier.add(Conv2D(64, (7, 7), activation='relu', input_shape=(128, 128, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second block
classifier.add(Conv2D(64, (1, 1), activation='relu'))
classifier.add(Conv2D(64, (1, 1), activation='relu'))
classifier.add(Conv2D(256, (1, 1), activation='relu'))

# Third block
classifier.add(Conv2D(128, (1, 1), activation='relu'))
classifier.add(Conv2D(128, (3, 3), activation='relu'))

# Fourth block
classifier.add(Conv2D(256, (1, 1), activation='relu'))
classifier.add(Conv2D(256, (3, 3), activation='relu'))

# Fifth block
classifier.add(Conv2D(512, (1, 1), activation='relu'))
classifier.add(Conv2D(512, (1, 1), activation='relu'))

# Sixth block
classifier.add(Conv2D(2048, (1, 1), activation='relu'))

classifier.add(GlobalAveragePooling2D())

# Hidden layer
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))

# Output layer (assuming 10 classes)
classifier.add(Dense(10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training and validation datasets
training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'val',
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical')

# Get the count of training and testing files
num_training_files = training_set.n
num_testing_files = test_set.n
print(f'Number of training files: {num_training_files}')
print(f'Number of testing files: {num_testing_files}')

# Calculate steps per epoch and validation steps
steps_per_epoch = num_training_files // 8
validation_steps = num_testing_files // 8

# Model checkpoint to save the best model
checkpoint = ModelCheckpoint('model_checkpoint.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min')

# Training the model
classifier.fit_generator(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=[checkpoint])
