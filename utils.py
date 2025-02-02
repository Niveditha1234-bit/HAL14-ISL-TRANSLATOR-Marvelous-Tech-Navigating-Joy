import cv2

def display_text(frame, text, position=(50, 50), color=(0, 255, 0)):
    """Display text on the frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
