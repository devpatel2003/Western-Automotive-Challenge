import os
from collections import defaultdict

# Define source dataset paths
data_dir = "data"
train_labels_path = os.path.join(data_dir, "train_labels.txt")
val_labels_path = os.path.join(data_dir, "val_labels.txt")
train_output_file = "lot_image_labels_train.txt"
val_output_file = "lot_image_labels_val.txt"

# Define one-hot encoding for lot classification
LOT_LABELS = {
    100: [1, 0, 0],  # Lot A
    40: [0, 1, 0],   # Lot B
    28: [0, 0, 1]    # Lot C
}

# Function to classify images based on bounding box count
def classify_lot(label_file):
    image_labels = {}
    with open(label_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Skipping invalid line: {line}")
                continue
            image_file = parts[0]
            image_labels.setdefault(image_file, []).append(parts[1:])
    return image_labels

# Process training and validation sets
train_lot = classify_lot(train_labels_path)
val_lot = classify_lot(val_labels_path)

# Function to generate lot mapping and save to a file
def save_lot_mapping(image_labels, output_file):
    lot_mapping = []
    for image, bboxes in image_labels.items():
        bbox_count = len(bboxes)
        lot_encoding = LOT_LABELS.get(bbox_count, [0, 0, 0])  # Default to unknown class [0, 0, 0]
        lot_mapping.append(f"{image} {' '.join(map(str, lot_encoding))}")

    # Save to file
    with open(output_file, "w") as f:
        f.write("\n".join(lot_mapping))
    print(f"✅ Lot image labels saved in {output_file}")

# Save separate label files for training and validation
save_lot_mapping(train_lot, train_output_file)
save_lot_mapping(val_lot, val_output_file)
