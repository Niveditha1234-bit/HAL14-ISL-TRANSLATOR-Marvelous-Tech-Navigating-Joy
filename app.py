from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("C:/HAL/ISL/sign_model.h5")

# Define labels (Make sure they match your training classes)
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_image(image):
    """Preprocess the image to match the model's expected input shape."""
    img = cv2.imread(image)
    img = cv2.resize(img, (64, 64))  # Resize to match model input shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if needed)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file temporarily
        image_path = os.path.join("uploads", file.filename)
        file.save(image_path)

        # Preprocess image and make prediction
        processed_img = preprocess_image(image_path)
        prediction = model.predict(processed_img)

        # Get the predicted label and confidence
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        confidence = float(np.max(prediction))

        # Return the prediction result
        return jsonify({'predicted_label': predicted_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
