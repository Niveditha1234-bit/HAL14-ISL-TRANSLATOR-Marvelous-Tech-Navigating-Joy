import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(hand_landmarks):
    """Extract (x, y, z) coordinates from hand landmarks."""
    keypoints = []
    for landmark in hand_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)

def get_hand_landmarks(frame_rgb):
    """Detect hand landmarks in a given frame."""
    results = hands.process(frame_rgb)
    return results.multi_hand_landmarks

import cv2
import numpy as np

def extract_hand_image(frame, hand_landmarks):
    """Crop and resize hand image to match model input (64x64)."""
    x_min = min([lm.x for lm in hand_landmarks.landmark])
    y_min = min([lm.y for lm in hand_landmarks.landmark])
    x_max = max([lm.x for lm in hand_landmarks.landmark])
    y_max = max([lm.y for lm in hand_landmarks.landmark])

    h, w, _ = frame.shape
    x_min, x_max = int(x_min * w), int(x_max * w)
    y_min, y_max = int(y_min * h), int(y_max * h)

    hand_crop = frame[y_min:y_max, x_min:x_max]
    hand_crop = cv2.resize(hand_crop, (64, 64))  # Resize to match CNN input
    hand_crop = hand_crop / 255.0  # Normalize

    return hand_crop
