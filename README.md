
# Emotion Classifier CNN

A Convolutional Neural Network designed for facial emotion recognition, capable of classifying emotions into 7 distinct categories from 48×48 grayscale facial images.

## Architecture Overview

![alt text](image.png)

The EmotionClassifier is a deep CNN architecture that follows a traditional pattern of progressive feature extraction through convolutional layers, followed by classification through fully connected layers.

### Network Flow

```
Input (1×48×48) → Conv Block 1 → Conv Block 2 → Conv Block 3 → Conv Block 4 → Flatten → FC Layers → Output (7 classes)
```

## Detailed Architecture

### Convolutional Blocks

The network consists of four convolutional blocks, each following the same pattern:
- **Conv2d** with 3×3 kernel, stride=1, padding=1
- **BatchNorm2d** for training stability
- **ReLU** activation function
- **MaxPool2d** with 2×2 kernel for spatial downsampling

#### Block 1: Input Processing
- **Input**: 1×48×48 (grayscale image)
- **Conv2d**: 1 → 32 channels
- **Output**: 32×24×24

#### Block 2: Feature Extraction
- **Input**: 32×24×24
- **Conv2d**: 32 → 64 channels
- **Output**: 64×12×12

#### Block 3: Deep Feature Learning
- **Input**: 64×12×12
- **Conv2d**: 64 → 128 channels
- **Output**: 128×6×6

#### Block 4: High-Level Features
- **Input**: 128×6×6
- **Conv2d**: 128 → 256 channels
- **Output**: 256×3×3

### Fully Connected Block

After the convolutional layers, the feature maps are flattened and passed through:

1. **Flatten Layer**: 256×3×3 → 2304 dimensional vector
2. **Linear Layer 1**: 2304 → 512 neurons
3. **ReLU Activation**
4. **Dropout**: 50% dropout rate for regularization
5. **Linear Layer 2**: 512 → 7 output classes

## Parameter Count

### Convolutional Layers

| Layer | Input Channels | Output Channels | Kernel Size | Parameters |
|-------|----------------|-----------------|-------------|------------|
| Conv1 | 1 | 32 | 3×3 | (1×3×3+1)×32 = **320** |
| Conv2 | 32 | 64 | 3×3 | (32×3×3+1)×64 = **18,496** |
| Conv3 | 64 | 128 | 3×3 | (64×3×3+1)×128 = **73,856** |
| Conv4 | 128 | 256 | 3×3 | (128×3×3+1)×256 = **295,168** |

### Batch Normalization Layers

| Layer | Channels | Parameters |
|-------|----------|------------|
| BN1 | 32 | 32×2 = **64** |
| BN2 | 64 | 64×2 = **128** |
| BN3 | 128 | 128×2 = **256** |
| BN4 | 256 | 256×2 = **512** |

### Fully Connected Layers

| Layer | Input Size | Output Size | Parameters |
|-------|------------|-------------|------------|
| FC1 | 2304 | 512 | (2304+1)×512 = **1,180,160** |
| FC2 | 512 | 7 | (512+1)×7 = **3,591** |

### Total Parameters

| Component | Parameters |
|-----------|------------|
| Convolutional Layers | 387,840 |
| Batch Normalization | 960 |
| Fully Connected Layers | 1,183,751 |
| **TOTAL** | **1,572,551** |

## Key Features

- **Progressive Channel Expansion**: 1 → 32 → 64 → 128 → 256 channels
- **Spatial Dimension Reduction**: 48×48 → 24×24 → 12×12 → 6×6 → 3×3
- **Regularization**: Batch normalization and dropout (0.5) to prevent overfitting
- **Efficient Design**: Uses 3×3 convolutions with padding to maintain spatial information
- **Hierarchical Learning**: Each layer learns increasingly complex features

## Input/Output Specifications

- **Input**: Single-channel (grayscale) images of size 48×48 pixels
- **Output**: 7-dimensional vector representing emotion class probabilities
- **Expected Classes**: Typically anger, disgust, fear, happiness, sadness, surprise, and neutral

## Usage

```python
import torch
import torch.nn as nn

# Initialize the model
model = EmotionClassifier(num_classes=7)

# Example forward pass
input_tensor = torch.randn(1, 1, 48, 48)  # Batch size 1
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # torch.Size([1, 7])
```

## Model Complexity

The model has approximately **1.57 million parameters**, making it a moderately-sized CNN suitable for:
- Training on limited computational resources
- Real-time inference applications
- Transfer learning scenarios
- Educational purposes for understanding CNN architectures

The architecture strikes a balance between model capacity and computational efficiency, making it well-suited for facial emotion recognition tasks.