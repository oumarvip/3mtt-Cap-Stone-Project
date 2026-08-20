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
    "cassava": os.path.join(ROOT_DIR, "models", "cassava_model.tflite"),
    "maize": os.path.join(ROOT_DIR, "models", "maize_model.tflite"),
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

def _load_model(model_path: str):
    print(f"\nLoading model from:")
    print(model_path)

    if not os.path.exists(model_path):
        print("❌ MODEL FILE DOES NOT EXIST")
        return None

    print("✅ Model file exists")
    print(f"File size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        print("✅ TFLite model loaded successfully")

        print("Input details:")
        print(interpreter.get_input_details())

        print("Output details:")
        print(interpreter.get_output_details())

        return interpreter

    except Exception as e:
        print("❌ FAILED TO LOAD TFLITE MODEL")
        print(type(e).__name__)
        print(e)

        return None






MODELS = {}


def get_model(crop: str):
    """
    Load only the requested TFLite model.
    """
    if crop not in MODEL_FILES:
        raise ValueError(f"Unsupported crop: {crop}")

    # Reuse model if already loaded
    if crop in MODELS and MODELS[crop] is not None:
        return MODELS[crop]

    # Remove previously loaded model
    MODELS.clear()

    # Load the requested TFLite model
    interpreter = _load_model(MODEL_FILES[crop])

    if interpreter is None:
        raise RuntimeError(f"{crop} model could not be loaded")

    MODELS[crop] = interpreter

    return interpreter




def preprocess_image(file_stream: io.BytesIO, target_size: tuple = (224, 224)) -> np.ndarray:
    image = Image.open(file_stream)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32")
    return arr  



def predict(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Add batch dimension
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype(np.float32)

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    probabilities = np.squeeze(output)

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index] * 100)

    return index, confidence




