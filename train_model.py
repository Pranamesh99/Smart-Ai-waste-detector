import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Define dataset path
dataset_path = "dataset"

# Load dataset dynamically
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


train_data = datagen.flow_from_directory(
    dataset_path, target_size=(150, 150), batch_size=32, 
    class_mode='categorical', subset="training")

val_data = datagen.flow_from_directory(
    dataset_path, target_size=(150, 150), batch_size=32, 
    class_mode='categorical', subset="validation")

# Get class labels dynamically
class_labels = list(train_data.class_indices.keys())
print("Detected Classes:", class_labels)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # Adjusted for dynamic classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save Model
model.save("waste_classifier.h5")
print("Model Saved Successfully!")