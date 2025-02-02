import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load trained LSTM model
model = tf.keras.models.load_model("C:/HAL/LSTM_ISL/lstm_model.h5")

# Define labels for prediction
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

def extract_hand_image(frame, hand_landmarks):
    """Crop, convert to grayscale, and resize hand image to (64, 64, 1)."""
    x_min = min([lm.x for lm in hand_landmarks.landmark])
    y_min = min([lm.y for lm in hand_landmarks.landmark])
    x_max = max([lm.x for lm in hand_landmarks.landmark])
    y_max = max([lm.y for lm in hand_landmarks.landmark])

    h, w, _ = frame.shape
    x_min, x_max = int(x_min * w), int(x_max * w)
    y_min, y_max = int(y_min * h), int(y_max * h)

    # Prevent invalid cropping
    if x_min >= x_max or y_min >= y_max:
        return None

    hand_crop = frame[y_min:y_max, x_min:x_max]

    # Ensure the cropped image is valid
    if hand_crop.size == 0:
        return None

    # Convert to grayscale
    hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

    # Resize to (64, 64)
    hand_crop = cv2.resize(hand_crop, (64, 64))

    # Normalize
    hand_crop = hand_crop / 255.0

    # Expand dimensions to match model input shape (1, 64, 64, 1)
    hand_crop = np.expand_dims(hand_crop, axis=-1)
    
    return hand_crop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predictions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_img = extract_hand_image(frame, hand_landmarks)

            if hand_img is not None:
                input_data = np.expand_dims(hand_img, axis=0)  # Shape: (1, 64, 64, 1)

                print("Input shape before prediction:", input_data.shape)

                res = model.predict(input_data)[0]
                print("Raw Predictions:", res)  # Debugging

                predicted_index = np.argmax(res)
                predicted_confidence = res[predicted_index]

                # Apply confidence threshold (e.g., 50%)
                if predicted_confidence >= 0.5:
                    predicted_sign = labels[predicted_index]
                    predictions.append(predicted_sign)
                    print(f"Predicted: {predicted_sign} (Confidence: {predicted_confidence:.2f})")
                else:
                    print("Low confidence, skipping prediction.")

    if predictions:
        predicted_text = " + ".join(predictions)  # Combine multiple hand predictions
        cv2.putText(frame, f'Sign: {predicted_text}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Translator", frame)
    
    if hand_img is not None:
        cv2.imshow("Hand Crop", hand_img)  # Show cropped image for debugging
        cv2.waitKey(1)  # Ensure the image updates

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
