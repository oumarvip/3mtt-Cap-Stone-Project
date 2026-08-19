from typing import Dict, List, Optional
import os
import numpy as np
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_FILES: Dict[str, str] = {
    "cassava": os.path.join(ROOT_DIR, "models", "cassava_model_finetune.keras"),
    "maize": os.path.join(ROOT_DIR, "models", "maize_finetune.keras"),
}

CLASS_NAMES: Dict[str, List[str]] = {
    "cassava": [
        "Cassava___bacterial_blight",
        "Cassava___brown_streak_disease",
        "Cassava___green_mottle",
        "Cassava___healthy",
        "Cassava___mosaic_disease",
    ],
    "maize": [
        "Blight",
        "Common_Rust",
        "Gray_Leaf_Spot",
        "Healthy",
    ],
}

DISPLAY_NAMES: Dict[str, str] = {
    "Cassava___bacterial_blight": "Cassava Bacterial Blight",
    "Cassava___brown_streak_disease": "Cassava Brown Streak Disease",
    "Cassava___green_mottle": "Cassava Green Mottle",
    "Cassava___healthy": "Healthy Cassava",
    "Cassava___mosaic_disease": "Cassava Mosaic Disease",
    "Blight": "Maize Blight",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Healthy": "Healthy Maize",
}

RECOMMENDATIONS: Dict[str, str] = {
    "Cassava___bacterial_blight": (
        "Remove and destroy infected plants; practice crop rotation "
        "and use clean planting material."
    ),

    "Cassava___brown_streak_disease": (
        "Use resistant varieties and remove infected plants; "
        "control whitefly vectors."
    ),

    "Cassava___green_mottle": (
        "Rogue infected plants and use certified disease-free cuttings; "
        "monitor fields regularly."
    ),

    "Cassava___healthy": (
        "No treatment needed. Maintain good agronomic practices "
        "to prevent disease."
    ),

    "Cassava___mosaic_disease": (
        "Use resistant cassava varieties and remove infected plants "
        "immediately to reduce disease spread."
    ),

        # Maize

    "Blight": (
        "Use resistant maize varieties and remove infected plant debris "
        "to reduce disease spread."
    ),

    "Common_Rust": (
        "Use resistant varieties and monitor the crop regularly. "
        "Where appropriate, apply recommended fungicide treatments early."
    ),

    "Gray_Leaf_Spot": (
        "Rotate crops, remove infected crop residue, and use resistant "
        "maize varieties to reduce disease pressure."
    ),

    "Healthy": (
        "No treatment needed. Continue good field management practices "
        "and regularly monitor the crop for signs of disease."
    ),
}




def _load_model(model_path: str) -> Optional[tf.keras.Model]:
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
        return load_model(model_path, custom_objects={"preprocess_input": mobilenet_preprocess})
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None




MODELS: Dict[str, Optional[tf.keras.Model]] = {}


def get_model(crop: str) -> Optional[tf.keras.Model]:
    """
    Load only the requested crop model.

    If another crop model is already loaded, unload it first
    to reduce memory usage.
    """
    if crop not in MODEL_FILES:
        raise ValueError(f"Unsupported crop: {crop}")

    # If the requested model is already loaded, reuse it
    if crop in MODELS and MODELS[crop] is not None:
        return MODELS[crop]

    # Remove any previously loaded model
    MODELS.clear()

    # Help Python/TensorFlow release unused memory
    tf.keras.backend.clear_session()

    # Load only the requested model
    model = _load_model(MODEL_FILES[crop])

    if model is None:
        raise RuntimeError(f"{crop} model could not be loaded")

    MODELS[crop] = model

    return model


def preprocess_image(file_stream: io.BytesIO, target_size: tuple = (224, 224)) -> np.ndarray:
    image = Image.open(file_stream)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32")
    return arr  


def predict(model: tf.keras.Model, image_array: np.ndarray, crop: str = "cassava") -> Dict[str, object]:
    if model is None:
        raise ValueError(f"{crop} model is not loaded")

    batch = np.expand_dims(image_array, axis=0)
    preds = model.predict(batch)
    probs = np.asarray(preds).squeeze()
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100.0)
    class_name = CLASS_NAMES[crop][idx] if idx < len(CLASS_NAMES[crop]) else "unknown"
    return {
        "index": idx,
        "class_name": class_name,
        "confidence": confidence,
    }
