# AgriScan: AI Leaf Disease Detector

AgriScan is an AI-powered web application for detecting diseases in cassava and maize leaves from uploaded images.

The system uses deep learning to analyze a leaf image, predict the most likely disease, provide a confidence score, and display recommended next steps.

> AgriScan is an assistive tool and should not replace professional agricultural diagnosis.

## Problem Statement

Crop diseases can significantly reduce agricultural productivity, particularly when farmers cannot identify diseases early.

AgriScan aims to provide a simple digital tool that can assist with preliminary identification of cassava and maize leaf diseases using computer vision.

## Objectives

The main objectives of AgriScan are to:

- Detect diseases from cassava and maize leaf images.
- Provide a predicted disease class.
- Display the model's confidence score.
- Provide basic recommended next steps.
- Provide a simple interface for users.
- Demonstrate the practical application of deep learning and computer vision in agriculture.

## Features

- Image upload
- Leaf image preview
- AI-based disease prediction
- Confidence score
- Disease-specific recommendations
- Responsive React interface
- Flask REST API
- TensorFlow/Keras inference
- Support for cassava and maize disease models

## Supported Crops

### Cassava

The cassava model contains five classes:

1. Cassava Bacterial Blight
2. Cassava Brown Streak Disease
3. Cassava Green Mottle
4. Cassava Healthy
5. Cassava Mosaic Disease

### Maize

The maize model contains four classes:

1. Blight
2. Common Rust
3. Gray Leaf Spot
4. Healthy

## How It Works

```text
User uploads leaf image
        ↓
Select the crop      
        ↓
React frontend
        ↓
Flask REST API
        ↓
Image preprocessing
        ↓
MobileNetV2 model
        ↓
Disease prediction
        ↓
Confidence score
        ↓
Recommendation
        ↓
Result displayed to user

# Maize Model

Before Fine-Tuning

The classification report was:

| Class                | Precision |   Recall | F1-Score | Support |
| -------------------- | --------: | -------: | -------: | ------: |
| Blight               |      0.87 |     0.90 |     0.88 |     225 |
| Common Rust          |      0.96 |     0.96 |     0.96 |     244 |
| Gray Leaf Spot       |      0.80 |     0.75 |     0.77 |     126 |
| Healthy              |      1.00 |     1.00 |     1.00 |     242 |
| Accuracy             |           |          |     0.92 |     837 |
| Macro Average        |    0.91   |   0.90   |     0.90 |     837 |
| Weighted Average     |    0.92   |   0.92   |     0.92 |     837 |

Blight          → 202
Common Rust     → 235
Gray Leaf Spot  → 94
Healthy         → 241

After Fine-Tuning

The fine-tuned model produced:

| Class                | Precision |   Recall | F1-Score | Support |
| -------------------- | --------: | -------: | -------: | ------: |
| Blight               |      0.92 |     0.80 |     0.86 |     225 |
| Common Rust          |      0.95 |     0.96 |     0.96 |     244 |
| Gray Leaf Spot       |      0.72 |     0.87 |     0.78 |     126 |
| Healthy              |      1.00 |     1.00 |     1.00 |     242 |
| Accuracy             |           |          |     0.92 |     837 |
| Macro Average        |      0.90 |     0.91 |     0.90 |     837 |
| Weighted Average     |      0.92 |     0.92 |     0.92 |     837 |

The fine-tuned model maintained approximately 92% accuracy while improving recall for Gray Leaf Spot from 0.75 to 0.87.

The model performed particularly well on:

Common Rust
Healthy

Gray Leaf Spot remained the most challenging class, although its recall improved after fine-tuning...


### Cassava Model

Cassava Model

The Cassava MobileNetV2 model was evaluated before and after fine-tuning.

Before Fine-Tuning

The initial model achieved approximately 66% accuracy.

| Class                | Precision |   Recall | F1-Score |  Support |
| -------------------- | --------: | -------: | -------: | -------: |
| Bacterial Blight     |      0.47 |     0.38 |     0.42 |      216 |
| Brown Streak Disease |      0.55 |     0.41 |     0.47 |      431 |
| Green Mottle         |      0.67 |     0.21 |     0.32 |      491 |
| Healthy              |      0.31 |     0.84 |     0.45 |      495 |
| Mosaic Disease       |      0.90 |     0.77 |     0.83 |     2646 |
| **Accuracy**         |           |          | **0.66** | **4279** |
| **Macro Average**    |  **0.58** | **0.52** | **0.50** | **4279** |
| **Weighted Average** |  **0.75** | **0.66** | **0.67** | **4279** |

The confusion matrix showed the following correct classifications:

Bacterial Blight       → 82
Brown Streak Disease   → 178
Green Mottle           → 104
Healthy                → 417
Mosaic Disease         → 2089

After Fine-Tuning

The fine-tuned Cassava model achieved approximately 73% accuracy.

| Class                | Precision |   Recall | F1-Score |  Support |
| -------------------- | --------: | -------: | -------: | -------: |
| Bacterial Blight     |      0.63 |     0.30 |     0.40 |      216 |
| Brown Streak Disease |      0.57 |     0.48 |     0.52 |      431 |
| Green Mottle         |      0.75 |     0.34 |     0.46 |      491 |
| Healthy              |      0.38 |     0.83 |     0.52 |      495 |
| Mosaic Disease       |      0.91 |     0.86 |     0.88 |     2646 |
| **Accuracy**         |           |          | **0.73** | **4279** |
| **Macro Average**    |  **0.65** | **0.56** | **0.56** | **4279** |
| **Weighted Average** |  **0.78** | **0.73** | **0.73** | **4279** |

Cassava Model Interpretation

The model achieved 73% overall accuracy, but the class-level results reveal the effect of dataset imbalance.

Cassava Mosaic Disease performed best, with:

Precision: 0.91
Recall: 0.86
F1-score: 0.88
Support: 2,646 images

The Healthy class achieved a high recall of 0.83, but its precision was only 0.38, indicating that many images predicted as healthy actually belonged to other classes.

The model struggled particularly with:

Bacterial Blight — recall: 0.30
Green Mottle — recall: 0.34
Brown Streak Disease — recall: 0.48

The macro F1-score of 0.56 is substantially lower than the weighted F1-score of 0.73. This indicates that model performance varies considerably between classes and that the larger classes have a stronger influence on the overall performance.

The Cassava dataset showed evidence of class imbalance, which affected the model's ability to recognize some disease classes equally well.
                 System Architcture
                ┌───────────────────┐
                │   User / Browser  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ React Frontend    │
                │ Image Upload UI   │
                └─────────┬─────────┘
                          │
                     HTTP Request
                          │
                          ▼
                ┌───────────────────┐
                │ Flask REST API    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Image Processing  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ MobileNetV2       │
                │ TensorFlow/Keras  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Prediction        │
                │ Confidence        │
                │ Recommendation    │
                └───────────────────┘

## AI-Assisted Development

AI tools were used as development assistants during this project.

In particular, AI assistance was used to help generate and structure parts of the React frontend UI and CSS, while the project architecture, machine learning workflow, model training, evaluation, Flask API integration, testing, and final implementation were reviewed, modified, and integrated as part of the project development process.

The AI-generated components were tested and adapted to meet the requirements of the AgriScan system.


                Technologies Used

Machine Learning
Python
TensorFlow
Keras
NumPy
scikit-learn
Backend
Flask
Flask-CORS
Python
Frontend
React
Vite
JavaScript
CSS
Development Tools
VS Code
UV
Jupyter
Git
GitHub


Project Structure

maize_cassava_disease_detector/
│
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── model.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── models/
│   ├── cassava_model.keras
│   ├── cassava_model_finetune.keras
│   ├── maize_model_1.keras
│   └── maize_finetune.keras
│
├── tests/
│
├── README.md
├── pyproject.toml
└── uv.lock