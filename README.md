# Simplified Hybrid CNN–Transformer Architecture

## Project Description

This project implements a simplified Hybrid CNN–Transformer architecture for image feature processing using the CIFAR-10 dataset.

The CNN extracts local spatial features from the input image, while the Transformer uses self-attention to learn relationships between different image regions.

## Architecture

Input Image
↓
CNN Feature Extraction
↓
Feature Map
↓
Tokenization
↓
Transformer Encoder
↓
Global Feature Extraction
↓
Classification
↓
Final Prediction

## Dataset

CIFAR-10 dataset containing 10 image classes.

- Training Images: 50,000
- Testing Images: 10,000
- Image Size: 32 × 32 × 3
- Number of Classes: 10

## Technologies Used

- Python 3.11
- TensorFlow 2.21.0
- NumPy
- SciPy
- Matplotlib
- Pillow
- Scikit-learn

## Model Details

### CNN
The CNN extracts local image features using convolution and pooling layers.

### Transformer
The CNN feature map is converted into 64 tokens. Multi-Head Self-Attention is then applied to learn relationships between image regions.

### Classification
Global average pooling is applied to the Transformer output, followed by a Dense layer with Softmax activation for 10-class classification.

## Results

Final Test Accuracy: 65.85%

Final Test Loss: 0.9767

## How to Run

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt