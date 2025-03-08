from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model("lot_classifier.h5")

# Predict function
def predict_lot(image_path):
    IMG_SIZE = 224
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)[0]
    lot_classes = ["Lot A", "Lot B", "Lot C"]
    return lot_classes[np.argmax(prediction)]

# Test an image
test_image = "more_test_data/test_images/2013-04-12_17_50_13.jpg"
predicted_lot = predict_lot(test_image)
print(f"Predicted Lot: {predicted_lot}")