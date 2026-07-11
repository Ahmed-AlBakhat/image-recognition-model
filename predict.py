from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_model.h5", compile=False)
print("Model loaded")

# Load labels
class_names = open("labels.txt", "r").readlines()

# Image path
image_path = "test2.jpg"
# Create array
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load and resize image
image = Image.open(image_path).convert("RGB")
print("Image loaded")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

# Convert image to array
image_array = np.asarray(image)

# Normalize
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load image into array
data[0] = normalized_image_array

# Predict
print("Predicting...")
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

print("Prediction:", class_name)
print("Confidence:", confidence_score)