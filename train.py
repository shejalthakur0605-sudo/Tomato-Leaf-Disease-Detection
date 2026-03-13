import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    'dataset/validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model and store history
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

print(history.history.keys())  

# Save model
model.save('tomato_disease_model.h5')
print("Model saved successfully!")

# -------- PLOTS -------- #

# Find correct validation accuracy key
if 'val_accuracy' in history.history:
    val_acc = history.history['val_accuracy']
elif 'val_binary_accuracy' in history.history:
    val_acc = history.history['val_binary_accuracy']
else:
    val_acc = None

# -------- Graph --------
import matplotlib.pyplot as plt

# Number of epochs
epochs = range(1, len(history.history['loss']) + 1)

# -------- Accuracy Graph --------
plt.figure()
plt.plot(epochs, history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_graph.png")
plt.close()

# -------- Loss Graph --------
plt.figure()
plt.plot(epochs, history.history['loss'], marker='o', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], marker='o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
plt.close()