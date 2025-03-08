from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("LotC_Training/yolov8n_lotc3/weights/best.pt")  # Adjust path if needed

# Load test image
image_path = ["more_test_data/test_images/2013-01-16_18_40_15.jpg"]  # Replace with your test image
results = model(image_path)  # Run inference
results = results[0]
# Count occupied and unoccupied spots
occupied_count = 0
unoccupied_count = 0

for box in results.boxes:
    class_id = int(box.cls[0].item())  # Get class ID
    if class_id == 0:  # Assuming 0 = Unoccupied, 1 = Occupied
        unoccupied_count += 1
    elif class_id == 1:
        occupied_count += 1

# Print results
print(f"Occupied spots: {occupied_count}")
print(f"Unoccupied spots: {unoccupied_count}")

# Show results
results.show()  # Display image with detections
results.save("out")  # Save output image