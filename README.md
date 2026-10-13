
## Assignment 2: CNN Image Classification

### CNN Architecture

This project implements a Convolutional Neural Network (CNN) using PyTorch.

Architecture:
- Input: 64 x 64 RGB image
- Conv2D: 16 filters, 3x3 kernel, stride 1, padding 1
- ReLU and MaxPooling2D (2x2, stride 2)
- Conv2D: 32 filters, 3x3 kernel, stride 1, padding 1
- ReLU and MaxPooling2D (2x2, stride 2)
- Flatten
- Fully connected layer: 100 units with ReLU
- Output layer: 10 classes

### Dataset and Training

Dataset: CIFAR10

The original 32x32 images are resized to 64x64 to match the CNN input.

Training configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 64
- Epochs: 3

Training loss:
- Epoch 1: 1.5350
- Epoch 2: 1.1887
- Epoch 3: 1.0521

Test accuracy: 62.82%

### Project Files

- app/cnn_model.py: CNN architecture
- train_cnn.py: CIFAR10 training and evaluation
- cnn_cifar10.pth: Trained CNN weights
- app/main.py: FastAPI application with image classification endpoint

### Setup

Install the dependencies:

```bash
python3 -m pip install "fastapi[standard]" python-multipart torch torchvision pillow spacy
python3 -m spacy download en_core_web_lg
python3 -m uvicorn app.main:app --reload
```

### Image Classification API

Endpoint: `POST /classify-image`

Upload an image using multipart/form-data.

The API returns:
- filename
- predicted_class
- confidence

Supported classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The original text embedding API remains available at `POST /embedding`.
