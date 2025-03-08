import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_dir = "data"
train_labels_file = "data/lot_image_labels_train.txt"
val_labels_file = "data/lot_image_labels_val.txt"
IMG_SIZE = 224  # Image size for CNN
BATCH_SIZE = 32

# Function to load image paths and labels
def load_labels_and_images(labels_file, image_folder):
    image_data = []
    labels = []
    with open(labels_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # Skip malformed lines
            image_name = parts[0]
            label = list(map(int, parts[1:]))  # Convert one-hot encoding to list
            image_path = os.path.join(image_folder, image_name)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)  # Read image
                img = cv2.resize(img, (224, 224))  # Resize to model input size
                img = img.astype(np.float32) / 255.0  # Normalize
                image_data.append(img)  # Store image array
                labels.append(label)  # Store corresponding label
    return np.array(image_data), np.array(labels)  # Convert to NumPy arrays

# Load training and validation data
# Load and preprocess training and validation data
train_images, train_labels = load_labels_and_images(train_labels_file, os.path.join(dataset_dir, "train_images"))
val_images, val_labels = load_labels_and_images(val_labels_file, os.path.join(dataset_dir, "val_images"))

# Verify Shapes Before Training
print(f"X_train shape: {train_images.shape}")  # Should be (N, 224, 224, 3)
print(f"y_train shape: {train_labels.shape}")  # Should be (N, 3)



# Apply preprocessing
X_train = train_images  # Ensure this has shape (N, 224, 224, 3)
X_val = val_images

# Convert one-hot encoded labels to numpy arrays
y_train = np.array(train_labels)
y_val = np.array(val_labels)

# Verify Shapes Before Training
print(f"X_train shape: {X_train.shape}")  # Should be (N, 224, 224, 3)
print(f"y_train shape: {y_train.shape}")  # Should be (N, 3)
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
model.fit(X_train, y_train, epochs=2, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose =1)

# Save Model
model.save("lot_classifier.h5")
print("✅ Model saved as lot_classifier.h5")

