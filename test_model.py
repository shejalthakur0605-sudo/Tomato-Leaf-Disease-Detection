import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("tomato_disease_model.h5")
print("Model loaded")

# Read image
img = cv2.imread("Healthy_Leaf.jpg")   # change image if needed
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.reshape(img, (1,224,224,3))

# Predict
prediction = model.predict(img)
value = prediction[0][0]

print("Raw Prediction Value:", value)

# Class labels
class_names = ["Early Blight", "Healthy"]

if value <= 0.5:
    print("Prediction:", class_names[0])
    print("Recommended Actions:")
    print("- Remove infected leaves")
    print("- Use Mancozeb or Copper fungicide")
    print("- Avoid overhead irrigation")
else:
    print("Prediction:", class_names[1])
    print("No action needed. Maintain good practices.")