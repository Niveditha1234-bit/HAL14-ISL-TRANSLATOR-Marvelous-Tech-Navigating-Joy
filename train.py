import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Reshape

# Set image dimensions
IMG_SIZE = 64

# Define directories
train_dir = "C:\HAL\LSTM_ISL\train"
classes = sorted(os.listdir(train_dir))  # Extract labels from folder names

# Load images
def load_images(data_dir):
    images, labels = [], []
    for label in classes:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(classes.index(label))  # Numeric encoding of labels
    return np.array(images), np.array(labels)

# Load data
X_train, y_train = load_images(train_dir)

# Normalize
X_train = X_train / 255.0
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Reshape((1, -1)),  # Reshape to feed into LSTM
    LSTM(64, return_sequences=False),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save model
model.save("lstm_model.h5")
