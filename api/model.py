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
    "cassava": os.path.join(ROOT_DIR, "cassava_model_finetune.keras"),  
    "maize": os.path.join(ROOT_DIR, "maize_finetune.keras"),  
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
    "Blight": "Use resistant maize varieties and remove infected debris to reduce spread.",
    "Common_Rust": "Apply fungicides early and improve air circulation where appropriate.",
    "Gray_Leaf_Spot": "Rotate crops and remove infected residue to limit disease pressure.",
    "Healthy": "No treatment needed. Maintain healthy field practices for maize.",
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


MODELS: Dict[str, Optional[tf.keras.Model]] = {
    crop: _load_model(path) for crop, path in MODEL_FILES.items()
}
MODEL = MODELS.get("cassava")


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
