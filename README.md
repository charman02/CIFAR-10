# CIFAR-10 Image Classification with Weighted Convolutional Blocks

This project implements a custom convolutional neural network (CNN) architecture for image classification on the CIFAR-10 dataset. The model is designed around a modular structure of intermediate blocks, where each block consists of multiple parallel convolutional layers whose outputs are combined via a learned weighted sum.

## Key Features
• Intermediate Blocks: Each block applies multiple convolutions in parallel, combines them using a learned softmax-weighted sum, and applies residual connections when needed.
• Custom Weight Computation: Weights for each convolution are dynamically computed from the input via global average pooling and a fully connected layer.
• Training Enhancements:
  • Dropout and batch normalization to improve generalization and training stability.
  • Residual connections with 1x1 convolutions to handle channel mismatches.
  • Max pooling after each block to reduce spatial dimensions.
• Training on CIFAR-10 using PyTorch with adjustable hyperparameters.

## Techniques and Hyperparameters
• Learning Rate: 0.001 (decays to 0.0001 after 30 epochs)
• Batch Size: 64
• Epochs: 50
• Dropout: 0.3 in intermediate blocks, 0.5 in output layers
• Activation: ReLU
• Optimizer: Adam
• Loss Function: Cross entropy loss
• Manual hyperparameter tuning via trial and error

## Results
The model achieves competitive testing accuracy (~87%) on CIFAR-10 with effective use of weighted feature aggregation and architectural enhancements like dropout, batch norm, and skip connections.

