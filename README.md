# MNIST Image Generation with DCGAN

This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating handwritten digits similar to those found in the MNIST dataset. The project demonstrates how to build, train, and utilize a GAN model for image generation.

## Step-by-Step Implementation Guide

### 1. Setup Environment
- **(Optional) Create and activate a virtual environment:**
  ```bash
  python -m venv gan_env
  source gan_env/bin/activate  # On Windows: gan_env\Scripts\activate

  # Install required packages
  pip install tensorflow numpy matplotlib
  
### 2. Load and Prepare the MNIST Dataset
  The MNIST dataset is automatically downloaded through TensorFlow's dataset API. The images are normalized to the range [-1, 1] to work well with the tanh activation function in the generator.

### 3. Build the Generator Network
  The generator transforms random noise into synthetic images:
  - Takes 100-dimensional noise vectors as input
  - Uses transposed convolutions to upsample from 7×7 to 28×28
  - Includes BatchNormalization and LeakyReLU for stable training
  - Outputs images with tanh activation

### 4. Build the Discriminator Network
  The discriminator distinguishes real from generated images:
  - Takes 28×28×1 images as input
  - Uses convolutional layers to downsample
  - Includes Dropout to prevent overfitting
  - Outputs a probability with sigmoid activation

### 5. Configure Training Process
  - Set batch size, learning rate, and number of epochs
  - Implement label smoothing (0.9 for real, 0.1 for fake)
  - Add small random noise to real images for training stability

### 6. Train the Model
  Run the training process for 2000 epochs, saving sample images every 200 epochs to monitor progress. The training saves both the generator and discriminator models after completion.
  
### 7. Generate Images with the Trained Model
  After training (or using a pre-trained model):
  - Generate a grid of synthetic digits
  - Experiment with different points in latent space
  - Create smooth interpolations between different digits

### 8. Visualize Results
  The code saves various visualizations to track progress and demonstrate results:
  - Training progression images at regular intervals
  - Final grid of generated digits
  - Latent space exploration samples
  - Interpolation between different digits

## Results

- Early epochs show noisy, unclear digit shapes
- Middle epochs show recognizable but blurry digits
- Final epochs produce clear, well-defined digits
- Latent space interpolations demonstrate smooth transitions between different digit styles

## Parameter Experimentation
  You can modify these parameters to see how they affect the generated images:

- *z_dim:* Dimension of the latent space (default: 100)
- *learning_rate:* Controls training speed and stability (default: 0.0002)
- *batch_size:* Number of samples per training batch (default: 128)
- *epochs:* Total training iterations (default: 2000)
- *Network architecture:* Adjust layer sizes and depths
