import os
import cv2

# Paths to YOLO dataset
dataset_dir = "datasets/LotC"
image_dirs = ["train/images", "val/images"]
label_dirs = ["train/labels", "val/labels"]
output_dir = "verify"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to load YOLO labels and draw bounding boxes
def verify_labels(image_dir, label_dir):
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(".jpg"):  # Ensure it's an image file
            continue
        
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):  # Skip if no label file exists
            continue

        # Load image
        image = cv2.imread(image_path)
        h, w, _ = image.shape  # Get original image dimensions

        # Read label file
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed labels

            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert from YOLO format to pixel coordinates
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)

            # Ensure bounding box is within image bounds
            x_min, x_max = max(0, x_min), min(w, x_max)
            y_min, y_max = max(0, y_min), min(h, y_max)

            # Draw bounding box on image
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for empty, Red for occupied
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, f"Class {int(class_id)}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save output image
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)

# Run verification for both train and val sets
for img_dir, lbl_dir in zip(image_dirs, label_dirs):
    verify_labels(os.path.join(dataset_dir, img_dir), os.path.join(dataset_dir, lbl_dir))

print(f"✅ Verification complete! Check the 'output/' folder for results.")
