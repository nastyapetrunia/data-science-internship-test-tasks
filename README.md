# Data Science Internship Test Tasks
![GitHub repo size](https://img.shields.io/github/repo-size/nastyapetrunia/data-science-internship-test-tasks)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/nastyapetrunia/data-science-internship-test-tasks)

Solutions for Data Science internship test tasks demonstrating machine learning, deep learning, and NLP capabilities.

## Overview

This repository contains two comprehensive ML projects:

### Task 1: MNIST Image Classification
A complete implementation comparing three different classification approaches for handwritten digit recognition:
- **Random Forest Classifier** (96.9% accuracy)
- **Feed-Forward Neural Network** (98.6% accuracy)
- **Convolutional Neural Network** (99.6% accuracy)

Built with OOP principles, unified interface, and Pydantic validation. See [task1_mnist_image_classification/README.md](task1_mnist_image_classification/README.md) for full details.

### Task 2: Named Entity Recognition + Image Classification Pipeline
A multimodal ML pipeline combining NER and computer vision to verify user claims about animals in images:
- **NER Model**: Fine-tuned MobileBERT for extracting animal entities from text
- **Image Classification**: EfficientNetB0 for 10-class animal classification (97% accuracy)
- **Verification Pipeline**: End-to-end system returning boolean verification results

See [task2_ner_image_classification/Readme.md](task2_ner_image_classification/Readme.md) for full details.

## Installation

### Prerequisites
- Python 3.9+ (tested on Python 3.12)
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/nastyapetrunia/data-science-internship-test-tasks.git
cd data-science-internship-test-tasks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for specific task
pip install -r task1_mnist_image_classification/requirements.txt
# or
pip install -r task2_ner_image_classification/requirements.txt
```

## Key Features

### Task 1 Highlights
- Object-oriented design with unified interface
- Comprehensive hyperparameter tuning and experiments
- Cross-model comparison with detailed metrics
- Type-safe configuration using Pydantic schemas
- Visualization of misclassified examples

### Task 2 Highlights
- Custom NER dataset with animal synonyms
- Transfer learning with EfficientNetB0
- Aspect ratio preservation with padding
- Class imbalance handling via augmentation
- End-to-end multimodal verification pipeline

## Technologies Used
- **ML/DL Frameworks**: TensorFlow, Keras, scikit-learn
- **NLP**: spaCy, Transformers (MobileBERT)
- **Data Handling**: NumPy, Pandas, Matplotlib
- **Validation**: Pydantic
- **Tools**: Jupyter, Kaggle API

## Author
Anastasiia Petrunia
