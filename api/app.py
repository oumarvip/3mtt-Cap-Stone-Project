from pathlib import Path
from typing import Any
import io
import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from . import model as model_module


ROOT_DIR = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "templates"),
        static_folder=str(ROOT_DIR / "static"),
    )

    CORS(app)

    @app.route("/", methods=["GET"])
    def index() -> Any:
        return render_template("index.html")

    @app.route("/api/predict", methods=["POST"])
    def predict_route() -> Any:

        # Get crop name
        crop = (request.form.get("crop") or "cassava").strip().lower()

        if crop not in model_module.MODEL_FILES:
            return jsonify({
                "success": False,
                "error": "Unsupported crop"
            }), 400

        # Load TFLite model
        try:
            model = model_module.get_model(crop)

        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"{crop} model not loaded: {e}"
            }), 500

        # Check image
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        # Preprocess image
        try:
            img_bytes = file.read()
            image = io.BytesIO(img_bytes)

            arr = model_module.preprocess_image(
                image,
                target_size=(224, 224)
            )

        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image file: {e}"
            }), 400

        # Make prediction
        try:
            prediction_index, confidence = model_module.predict(
                model,
                arr
            )

        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Prediction failed: {e}"
            }), 500

        # Convert prediction index to class name
        try:
            class_name = model_module.CLASS_NAMES[crop][prediction_index]

        except (IndexError, KeyError):
            return jsonify({
                "success": False,
                "error": "Invalid prediction class returned by model"
            }), 500

        # Get display information
        display_name = model_module.DISPLAY_NAMES.get(
            class_name,
            class_name
        )

        recommendation = model_module.RECOMMENDATIONS.get(
            class_name,
            "No recommendation available."
        )

        # Return result
        return jsonify({
            "success": True,
            "crop": crop,
            "class_name": class_name,
            "display_name": display_name,
            "confidence": round(float(confidence), 2),
            "recommendation": recommendation
        })

    return app


if __name__ == "__main__":
    app = create_app()

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )