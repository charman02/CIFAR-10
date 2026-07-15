# CIFAR-10 Image Classifier with Learned Weighted Convolutional Blocks

A custom convolutional neural network for image classification on the CIFAR-10 dataset, built in PyTorch. Instead of stacking fixed convolution layers, each block runs several convolutions in parallel and fuses their outputs using an **input-dependent, softmax-weighted sum** — a lightweight, attention-like feature-aggregation mechanism.

The recorded training run reaches a best test accuracy of **86.9%** on CIFAR-10.

> This project was originally built as an assignment for the ECS7026P *Neural Networks and Deep Learning* module (Queen Mary University of London). It is a single self-contained notebook.

## Tech Stack

- **Python** with **PyTorch** and **torchvision** (model, dataset loading, and augmentation)
- **NumPy** and **Matplotlib** (evaluation and training/loss plots)
- Runs on GPU (CUDA) when available, and falls back to CPU automatically

## Getting Started

The entire project lives in `cifar-10_image_classifier.ipynb`.

```bash
# Clone the repository
git clone https://github.com/charman02/cifar10-image-classifier.git
cd cifar10-image-classifier

# Install dependencies
pip install torch torchvision numpy matplotlib jupyter

# Launch the notebook
jupyter notebook cifar-10_image_classifier.ipynb
```

Run the cells top to bottom. CIFAR-10 (~170 MB) is downloaded automatically to `./data` on first run. Training uses the GPU if one is available and otherwise runs on CPU (much slower — a GPU is recommended for the full 50-epoch run). The best-performing model is checkpointed to `best_cifar10_model_simple.pth` whenever test accuracy improves.

## How It Works

### Data

CIFAR-10 (60,000 32x32 color images across 10 classes) is loaded via `torchvision`. Training images are augmented with random horizontal flips and random crops (4-pixel padding); both splits are normalized with the standard per-channel CIFAR-10 mean/std.

### The Intermediate Block

The core idea is the `IntermediateBlock`. Rather than applying a single convolution, each block applies `num_convs` separate 3x3 convolutions to the same input and then learns how much each one should contribute, *per input example*:

1. Apply each of the parallel convolutions to the input, producing `num_convs` feature maps.
2. Compute a summary of the input via global average pooling (mean over the spatial dimensions), giving one value per input channel.
3. Feed that summary through a fully connected layer and a `softmax` to produce a set of weights that sum to 1 — one weight per parallel convolution.
4. Combine the convolution outputs as a weighted sum using those weights.
5. Apply dropout (0.3) and batch normalization, add a residual/skip connection (with a 1x1 convolution when the channel count changes), and finish with a ReLU.

Because the weights are derived from the input itself, the block can emphasize different convolutional filters for different images — a form of learned, content-dependent feature selection.

### The Network

`Net` is configuration-driven. It takes a list of `(in_channels, out_channels, num_convs)` tuples and builds one `IntermediateBlock` per entry, so the depth and width of the model are controlled entirely by that list. The default configuration is:

| Block | In channels | Out channels | Parallel convs |
|-------|-------------|--------------|----------------|
| 1     | 64          | 128          | 2              |
| 2     | 128         | 256          | 2              |
| 3     | 256         | 512          | 3              |
| 4     | 512         | 768          | 3              |

The full forward pass is: an initial 3x3 convolution (+ batch norm + ReLU) → the four intermediate blocks (with 2x2 max pooling between blocks to reduce spatial size) → global average pooling → an output block of fully connected layers producing 10 class logits. Weights are initialized with Kaiming (He) initialization.

## Training

| Setting | Value |
|---------|-------|
| Optimizer | Adam, weight decay (L2) 1e-4 |
| Learning rate | 0.001, manually decayed to 0.0001 after epoch 30 |
| Batch size | 64 |
| Epochs | 50 |
| Loss | Cross-entropy |
| Dropout | 0.3 in blocks, 0.5 in the output FC layers |
| Regularization | Batch norm, dropout, data augmentation, weight decay |

Hyperparameters were tuned manually by adjusting one at a time and monitoring test performance. The model is checkpointed whenever test accuracy improves, and the notebook plots training/test accuracy and per-batch training loss at the end.

## Results

On the recorded 50-epoch run, the best checkpointed model reached **86.9% test accuracy** on CIFAR-10. Accuracy and loss curves are produced inline at the end of the notebook.

## License

Released under the [MIT License](LICENSE).
