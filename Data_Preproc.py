#!/usr/bin/env python3
"""
Data_Preproc.py
---------------
This script reads your training and validation label files:
  - Data/train_labels.txt
  - Data/val_labels.txt

Each line in these label files is of the form:
  image_file class_id xmin ymin xmax ymax

It then:
  1. Copies the corresponding images into a YOLO-style folder structure:
        datasets/train/images, datasets/train/labels
        datasets/val/images,   datasets/val/labels
  2. Creates one label file per image in YOLO format:
       class_id x_center y_center width height
     where all coordinates are normalized by the actual image width/height.

We ignore test data in this script. If you need to process test data,
you can replicate the logic below for your test_labels and test_images.
"""

import os
import shutil
import cv2
from collections import defaultdict

# ------------------------------------------------------------------------------
# 1. DEFINE SOURCE AND DESTINATION DIRECTORIES
# ------------------------------------------------------------------------------
DATA_DIR = "data"  # Top-level data folder

# Source directories
TRAIN_IMAGES_PATH = os.path.join(DATA_DIR, "train_images")   # Folder of training images
VAL_IMAGES_PATH   = os.path.join(DATA_DIR, "val_images")     # Folder of validation images

# Source label files
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train_labels.txt")
VAL_LABELS_FILE   = os.path.join(DATA_DIR, "val_labels.txt")

# Destination: YOLO-style dataset structure
DATASET_DIR  = "datasets_gen"        # Where we'll put the YOLO folders
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG_DIR   = os.path.join(DATASET_DIR, "val",   "images")
VAL_LBL_DIR   = os.path.join(DATASET_DIR, "val",   "labels")

# Ensure the YOLO directories exist
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LBL_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LBL_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# 2. PARSE LABEL FILES (NO FILTERING BY BOUNDING BOX COUNT)
# ------------------------------------------------------------------------------
def parse_labels(label_file):
    """
    Reads a label file of the form:
       image_file class_id xmin ymin xmax ymax
    and returns a dict: { image_filename: [(class_id, xmin, ymin, xmax, ymax), ...], ... }
    """
    image_labels = defaultdict(list)
    
    if not os.path.isfile(label_file):
        print(f"Warning: Label file not found: {label_file}")
        return image_labels
    
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            
            image_file = parts[0]
            try:
                # Convert to 0-based class index for YOLO (class_id - 1)
                # If your classes are already 0-based, remove the "- 1"
                class_id = int(parts[1]) - 1
                xmin = float(parts[2])
                ymin = float(parts[3])
                xmax = float(parts[4])
                ymax = float(parts[5])
                
                image_labels[image_file].append((class_id, xmin, ymin, xmax, ymax))
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")
                continue
    
    return image_labels


# ------------------------------------------------------------------------------
# 3. CONVERT BOUNDING BOXES TO YOLO FORMAT AND WRITE ONE LABEL FILE PER IMAGE
# ------------------------------------------------------------------------------
def save_data(image_labels, src_image_dir, dst_image_dir, dst_label_dir):
    """
    Copies each image from src_image_dir to dst_image_dir.
    Writes a YOLO-format label file for each image to dst_label_dir.
    
    YOLO format: class_id x_center y_center width height  (all normalized)
    """
    for image_filename, bboxes in image_labels.items():
        # Source image path
        src_img_path = os.path.join(src_image_dir, image_filename)
        
        # Destination image path
        dst_img_path = os.path.join(dst_image_dir, image_filename)
        
        if not os.path.exists(src_img_path):
            print(f"Warning: Image not found: {src_img_path}")
            continue
        
        # Copy the image to the YOLO dataset folder
        shutil.copy(src_img_path, dst_img_path)
        
        # Load the image to get its dimensions
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"Warning: Unable to read {src_img_path}. Skipping bounding boxes.")
            continue
        h, w = img.shape[:2]
        
        # Create the label file path (same name, but .txt)
        label_file = os.path.join(dst_label_dir, image_filename.replace(".jpg", ".txt"))
        
        # Write all bounding boxes for this image
        with open(label_file, "w") as f:
            for (class_id, xmin, ymin, xmax, ymax) in bboxes:
                # Convert to YOLO normalized coordinates
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                width    = (xmax - xmin) / w
                height   = (ymax - ymin) / h
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# ------------------------------------------------------------------------------
# 4. MAIN LOGIC
# ------------------------------------------------------------------------------
def main():
    # Parse the training and validation label files
    train_data = parse_labels(TRAIN_LABELS_FILE)
    val_data   = parse_labels(VAL_LABELS_FILE)
    
    # Save the training data (images + YOLO labels)
    save_data(
        image_labels=train_data,
        src_image_dir=TRAIN_IMAGES_PATH,
        dst_image_dir=TRAIN_IMG_DIR,
        dst_label_dir=TRAIN_LBL_DIR
    )
    
    # Save the validation data (images + YOLO labels)
    save_data(
        image_labels=val_data,
        src_image_dir=VAL_IMAGES_PATH,
        dst_image_dir=VAL_IMG_DIR,
        dst_label_dir=VAL_LBL_DIR
    )
    
    print("✅ YOLO dataset prepared successfully!")
    print(f"   - Training images/labels in: {os.path.join(DATASET_DIR, 'train')}")
    print(f"   - Validation images/labels in: {os.path.join(DATASET_DIR, 'val')}")


if __name__ == "__main__":
    main()
