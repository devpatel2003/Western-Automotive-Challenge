import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

# Paths
test_images_dir = "more_test_data/test_images"
test_labels_file = "more_test_data/test_labels.txt"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Load the first 4 test images from the folder
test_images = sorted(os.listdir(test_images_dir))[:4]
test_images = [os.path.join(test_images_dir, img) for img in test_images]

# Load YOLO model
model = YOLO("LotA_Training/yolov8n_lot_a/weights/best.pt")

def yolo_to_minmax(pred_boxes, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to min-max format."""
    minmax_boxes = []
    for box in pred_boxes:
        x_center, y_center, w, h = box[:4]

        # Convert from YOLO (normalized) to min-max pixel format
        x_min = int((x_center - w / 2) * img_width)
        y_min = int((y_center - h / 2) * img_height)
        x_max = int((x_center + w / 2) * img_width)
        y_max = int((y_center + h / 2) * img_height)

        minmax_boxes.append([x_min, y_min, x_max, y_max])
    return minmax_boxes

def load_ground_truth_labels(labels_file, selected_images):
    """Load ground truth labels from test_labels.txt."""
    gt_labels = defaultdict(list)  # {image_name: [(class_id, x_min, y_min, x_max, y_max), ...]}
    
    with open(labels_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            image_name, class_id, x_min, y_min, x_max, y_max = parts
            class_id = int(class_id)
            
            # Convert to float (handles decimal values)
            box = (float(x_min), float(y_min), float(x_max), float(y_max), class_id)
            if image_name in selected_images:
                gt_labels[image_name].append(box)
    
    return gt_labels

# Get selected image names
selected_image_names = [os.path.basename(img) for img in test_images]
ground_truth = load_ground_truth_labels(test_labels_file, selected_image_names)

def draw_boxes(image, boxes, color, labels):
    """
    Draw bounding boxes with labels.
    :param image: Image to draw on
    :param boxes: List of (x_min, y_min, x_max, y_max, class_id)
    :param color: Color for boxes
    :param labels: Dictionary of class labels {class_id: "LabelName"}
    :return: Image with drawn boxes
    """
    for box in boxes:
        x_min, y_min, x_max, y_max, class_id = box
        class_name = labels.get(class_id, str(class_id))  # Get label name
        
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(image, class_name, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    return image

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute intersection
    x_left = max(x1, x1_p)
    y_top = max(y1, y1_p)
    x_right = min(x2, x2_p)
    y_bottom = min(y2, y2_p)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute areas
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # Compute union
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

# Class labels (modify if needed)
class_labels = {0: "Unoccupied", 1: "Occupied"}

csv_results = []
all_classes = set()

for image_path in test_images:
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)

    # Run YOLO inference
    results = model(image_path)
    results = results[0]  # Extract first result
    pred_boxes = results.boxes.data.cpu().numpy()

    # Convert YOLO (normalized) to min-max format
    img_height, img_width, _ = image.shape
    pred_boxes_min_max = yolo_to_minmax(pred_boxes, img_width, img_height)

    class_counts = defaultdict(int)

    # Extract class IDs for predictions
    pred_classes = [int(pred[5]) for pred in pred_boxes]

    # Convert predicted boxes to include class_id
    pred_boxes_for_drawing = [(x_min, y_min, x_max, y_max, cls) for (x_min, y_min, x_max, y_max), cls in zip(pred_boxes_min_max, pred_classes)]

    for cls in pred_classes:
        class_counts[cls] += 1
        all_classes.add(cls)

    # Load ground truth labels
    gt_boxes = ground_truth.get(image_name, [])

    # Count correct detections based on IoU ≥ 0.5
    correct_detections = 0
    for gt_class, gt_x_min, gt_y_min, gt_x_max, gt_y_max in gt_boxes:
        for pred_x_min, pred_y_min, pred_x_max, pred_y_max in pred_boxes_min_max:
            iou = compute_iou((gt_x_min, gt_y_min, gt_x_max, gt_y_max), 
                              (pred_x_min, pred_y_min, pred_x_max, pred_y_max))
            if iou >= 0.5:
                correct_detections += 1
                break  # One GT box should match only one predicted box

    # Draw bounding boxes
    image = draw_boxes(image, gt_boxes, (0, 0, 255), class_labels)  # Red for ground truth
    image = draw_boxes(image, pred_boxes_for_drawing, (0, 255, 0), class_labels)  # Green for predictions

    # Save annotated image
    output_image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_image_path, image)

    # Append results for CSV
    csv_results.append({
        "Image Name": image_name,
        "Correct Detections": correct_detections,
        "Total Predictions": len(pred_boxes_min_max),
        "Total Ground Truth": len(gt_boxes),
        **{f"Class_{cls}": class_counts.get(cls, 0) for cls in all_classes}
    })

# Save CSV results
csv_file = "detection_results.csv"
df = pd.DataFrame(csv_results)
df.to_csv(csv_file, index=False)

print(f"Results saved to {csv_file}")
print(f"Annotated images saved in {output_dir}/")
