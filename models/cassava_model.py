from typing import Dict, List, Optional
import os
import numpy as np
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Expected model filename (project root)
MODEL_FILENAME = os.path.join(os.path.dirname(__file__), "..", "cassava_disease_model.keras")


CLASS_NAMES: List[str] = [
    "Cassava___bacterial_blight",
    "Cassava___brown_streak_disease",
    "Cassava___green_mottle",
    "Cassava___healthy",
    "Cassava___mosaic_disease",
]


DISPLAY_NAMES: Dict[str, str] = {
    "Cassava___bacterial_blight": "Cassava Bacterial Blight",
    "Cassava___brown_streak_disease": "Cassava Brown Streak Disease",
    "Cassava___green_mottle": "Cassava Green Mottle",
    "Cassava___healthy": "Healthy Cassava",
    "Cassava___mosaic_disease": "Cassava Mosaic Disease",
}


RECOMMENDATIONS: Dict[str, str] = {
    "Cassava___bacterial_blight": (
        "Remove and destroy infected plants; practice crop rotation and use clean planting material."
    ),
    "Cassava___brown_streak_disease": (
        "Use resistant varieties and remove infected plants; control whitefly vectors."
    ),
    "Cassava___green_mottle": (
        "Rogue infected plants and use certified disease-free cuttings; monitor fields regularly."
    ),
    "Cassava___healthy": ("No treatment needed. Maintain good agronomic practices to prevent disease."),
    "Cassava___mosaic_disease": (
        "Use resistant cassava varieties (e.g., TME 419) and clear infected plants immediately to prevent whitefly transmission."
    ),
}


def _load_model() -> Optional[tf.keras.Model]:
    """Attempt to load the Keras model from disk. Returns None on failure."""
    try:
        model_path = os.path.abspath(MODEL_FILENAME)
        if not os.path.exists(model_path):
            return None
        model = load_model(model_path)
        return model
    except Exception:
        return None


# Load model at import time (best-effort). App should handle None model.
MODEL = _load_model()


def preprocess_image(file_stream: io.BytesIO, target_size: tuple = (224, 224)) -> np.ndarray:
    """Read image bytes, convert to RGB, resize, and preprocess for Keras model input.

    Returns a float32 numpy array of shape (224, 224, 3).
    """
    image = Image.open(file_stream)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32")
    # Use Keras imagenet-style preprocessing as a sensible default
    arr = preprocess_input(arr)
    return arr


def predict(model: tf.keras.Model, image_array: np.ndarray) -> Dict[str, object]:
    """Run inference and return structured prediction info."""
    if model is None:
        raise ValueError("Model is not loaded")

    # model expects batch dimension
    batch = np.expand_dims(image_array, axis=0)
    preds = model.predict(batch)
    # Ensure 1D array of probabilities
    probs = np.asarray(preds).squeeze()
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100.0)
    class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "unknown"
    return {
        "index": idx,
        "class_name": class_name,
        "confidence": confidence,
    }
