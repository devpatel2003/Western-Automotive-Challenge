from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Define datasets for Lot A, B, and C
    datasets = {
        #"LotA": "datasets/LotA/data.yaml",
        #"LotB": "datasets/LotB/data.yaml",
        "LotC": "datasets/LotC/data.yaml"
    }

    # Iterate over each dataset and train separately
    for lot, data_path in datasets.items():
        print(f"Training on {lot}...")

        # Load the pre-trained YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Train the model
        model.train(
            data=data_path,  # Path to dataset.yaml
            epochs=75,       # Number of training epochs
            imgsz=640,       # Input image size
            batch=32,        # Adjust batch size based on GPU memory
            device="cuda",   # Use GPU ("cuda") or CPU ("cpu")
            workers=0,       # Number of dataloader workers (adjust if needed)
            optimizer="Adam", # Use Adam optimizer (default: SGD)
            patience=10,      # Early stopping patience
            save=True,        # Save checkpoints
            project=f"{lot}_Training",  # Save training runs per lot
            name=f"yolov8n_{lot.lower()}", # Experiment name
            amp=True
        )

        # Save the best trained model and export to TFLite
        print(f"Exporting {lot} model to TFLite...")
        model.export(format="tflite", name=f"{lot}_trained_model.tflite")

    print("Training completed for Lot A, B, and C!")