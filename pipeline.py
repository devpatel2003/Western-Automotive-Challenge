import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load trained CNN model
cnn_model = load_model("lot_classifier.h5")

# Define paths for YOLO models for each parking lot
YOLO_MODELS = {
    "Lot A": "LotA_Training/yolov8n_lota/weights/best.pt",
    "Lot B": "LotB_Training/yolov8n_lotb/weights/best.pt",
    "Lot C": "LotC_Training/yolov8n_lotc/weights/best.pt"
}

# Predict function for lot classification
def predict_lot(image_path):
    IMG_SIZE = 224
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = cnn_model.predict(img)[0]
    lot_classes = ["Lot A", "Lot B", "Lot C"]
    return lot_classes[np.argmax(prediction)]

# Parking spot detection function
def detect_parking_spots(image_path, yolo_model_path):
    model = YOLO(yolo_model_path)  # Load YOLO model
    results = model(image_path)  # Run inference
    results = results[0]

    # Count occupied and unoccupied spots
    occupied_count = 0
    unoccupied_count = 0

    for box in results.boxes:
        class_id = int(box.cls[0].item())  # Get class ID
        if class_id == 0:  
            unoccupied_count += 1
        elif class_id == 1:
            occupied_count += 1

    # Draw bounding boxes
    image = cv2.imread(image_path)
    for box in results.boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for empty, Red for occupied
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, f"Class {class_id}", (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save output image
    output_image_path = "output.jpg"
    cv2.imwrite(output_image_path, image)

    print(f"Occupied spots: {occupied_count}")
    print(f"Unoccupied spots: {unoccupied_count}")
    print(f"Processed image saved as {output_image_path}")

# Full pipeline function
def process_image(image_path):

    # Step 1: Classify the parking lot
    predicted_lot = predict_lot(image_path)
    print(f"Predicted Lot: {predicted_lot}")

    # Step 2: Select the correct YOLO model
    yolo_model_path = YOLO_MODELS.get(predicted_lot)
    if not yolo_model_path:
        print("No model found for this lot.")
        return

    # Step 3: Detect parking spots using the correct YOLO model
    detect_parking_spots(image_path, yolo_model_path)


test_image = "more_test_data/test_images/2012-09-11_15_36_32.jpg"  # Change to your test image path
process_image(test_image)
