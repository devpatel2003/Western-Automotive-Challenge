import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_dir = "data"
train_labels_file = "lot_image_labels_train.txt"
val_labels_file = "lot_image_labels_val.txt"
IMG_SIZE = 224  # Image size for CNN
BATCH_SIZE = 32

# Function to load image paths and labels
def load_labels(labels_file):
    image_paths = []
    labels = []
    with open(labels_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # Skip malformed lines
            image_name = parts[0]
            label = list(map(int, parts[1:]))  # Convert one-hot encoding to list
            image_path = os.path.join(dataset_dir, "train_images", image_name)  # Adjust if needed
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(label)
    return np.array(image_paths), np.array(labels)

# Load training and validation data
train_images, train_labels = load_labels(train_labels_file)
val_images, val_labels = load_labels(val_labels_file)

# Function to preprocess images
def preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
        img = img.astype(np.float32) / 255.0  # Normalize
        images.append(img)
    return np.array(images)

# Preprocess images
X_train = preprocess_images(train_images)
X_val = preprocess_images(val_images)

# Convert one-hot encoded labels to categorical
y_train = np.array(train_labels)
y_val = np.array(val_labels)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 output classes (Lot A, B, C)
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose =1)

# Save Model
model.save("lot_classifier.h5")
print("✅ Model saved as lot_classifier.h5")

