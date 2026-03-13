import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("tomato_disease_model.h5")

img = cv2.imread("Early_Blight.jpg")   # put any leaf image here
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.reshape(img, (1,224,224,3))

prediction = model.predict(img)

print(prediction)

if prediction[0][0] <= 0.5:
    print("Early Blight Detected")
    print("Recommended Actions:")
    print("- Remove infected leaves")
    print("- Use Mancozeb or Copper fungicide")
    print("- Avoid overhead irrigation")
else:
    print("Healthy Leaf")
    print("No action needed. Maintain good practices.")