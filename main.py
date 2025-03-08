import cv2
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from tensorflow import keras
import argparse
import os
import time

# Model selection based on parking lot classification
LOT_MODELS = {
    "Lot_A": "lot_a_model.tflite",
    "Lot_B": "lot_b_model.tflite",
    "Lot_C": "lot_c_model.tflite",
}

# Function to classify parking lot type
def classify_lot(image_path, lot_classifier_model):
    model = keras.models.load_model(lot_classifier_model)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    lot_classes = ["Lot_A", "Lot_B", "Lot_C"]
    return lot_classes[np.argmax(prediction)]

# Load and run TFLite model
def run_tflite_model(tflite_model_path, image):
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = cv2.resize(image, (640, 640)) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to detect parking spots and classify occupancy
def detect_parking_spots(image_path, model_path):
    image = cv2.imread(image_path)
    detections = run_tflite_model(model_path, image)
    bounding_boxes = []
    occupancy_labels = []
    for det in detections[0][0]:
        if det[4] > 0.5:  # Confidence threshold
            x_min, y_min, x_max, y_max = map(int, det[:4])
            bounding_boxes.append((x_min, y_min, x_max, y_max))
            occupancy_labels.append("Occupied" if det[5] == 2 else "Unoccupied")
    return bounding_boxes, occupancy_labels

# Function to draw bounding boxes on image
def draw_detections(image_path, bounding_boxes, occupancy_labels):
    image = cv2.imread(image_path)
    for box, label in zip(bounding_boxes, occupancy_labels):
        x_min, y_min, x_max, y_max = map(int, box)
        color = (0, 0, 255) if label == "Occupied" else (0, 255, 0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("output.jpg", image)
    return "output.jpg"

# Main inference function
def main(image_path, lot_classifier_model):
    lot_type = classify_lot(image_path, lot_classifier_model)
    model_path = LOT_MODELS[lot_type]
    bounding_boxes, occupancy_labels = detect_parking_spots(image_path, model_path)
    output_image = draw_detections(image_path, bounding_boxes, occupancy_labels)
    print(f"Processed Image Saved: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the parking lot image")
    parser.add_argument("--lot_model", type=str, required=True, help="Path to the lot classification model")
    args = parser.parse_args()
    main(args.image, args.lot_model)