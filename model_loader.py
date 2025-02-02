from tensorflow.keras.models import load_model

def load_sign_model(model_path):
    """Load the trained LSTM model."""
    model = load_model("C:\HAL\LSTM_ISL\lstm_model.h5")
    return model
