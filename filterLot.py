import os
import shutil
from collections import defaultdict

# Define source dataset paths
data_dir = "data"
train_images_path = os.path.join(data_dir, "train_images")
val_images_path = os.path.join(data_dir, "val_images")
train_labels_path = os.path.join(data_dir, "train_labels.txt")
val_labels_path = os.path.join(data_dir, "val_labels.txt")

# Define YOLO dataset paths
dataset_dir = "datasets/LotC"
train_img_dir = os.path.join(dataset_dir, "train/images")
train_lbl_dir = os.path.join(dataset_dir, "train/labels")
val_img_dir = os.path.join(dataset_dir, "val/images")
val_lbl_dir = os.path.join(dataset_dir, "val/labels")

# Ensure directories exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Function to filter images based on bounding boxes
def filter_lot(label_file, bbox_count=28):
    image_labels = defaultdict(list)
    with open(label_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Skipping invalid line: {line}")
                continue
            image_file = parts[0]
            try:
                class_id = int(parts[1]) - 1  # Convert class_id to YOLO format (0-based index)
                x_min, x_max, y_min, y_max = map(float, parts[2:])  # Extract bounding box
                image_labels[image_file].append((class_id, x_min, x_max, y_min, y_max))
            except ValueError:
                print(f"Skipping malformed line: {line}")
                continue
    return {k: v for k, v in image_labels.items() if len(v) == bbox_count}

# Process training and validation sets
train_lot = filter_lot(train_labels_path)
val_lot = filter_lot(val_labels_path)

# Function to save images and labels in YOLO format
def save_data(image_labels, image_source_dir, image_dest_dir, label_dest_dir):
    for image, bboxes in image_labels.items():
        image_path = os.path.join(image_source_dir, image)
        image_name = os.path.basename(image)

        # Move image to YOLO dataset directory
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(image_dest_dir, image_name))

        # Create YOLO label file
        label_file = os.path.join(label_dest_dir, image_name.replace(".jpg", ".txt"))
        with open(label_file, "w") as f:
            for bbox in bboxes:
                class_id, x_min, y_min, x_max, y_max = bbox
                #print(bbox)

                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = (x_min + x_max) / 2.0 / 640  # Normalize (assuming 640x640 images)
                y_center = (y_min + y_max) / 2.0 / 640
                width = abs(x_max - x_min) / 640.0
                height = abs(y_max - y_min) / 640.0

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Save training and validation data
save_data(train_lot, train_images_path, train_img_dir, train_lbl_dir)
save_data(val_lot, val_images_path, val_img_dir, val_lbl_dir)

print(f"✅ YOLO dataset prepared in {dataset_dir}")
