# pytorch-cifar10-experiments

Image classification on CIFAR-10 using a custom CNN built with PyTorch.  
This repo contains training code, experiment results, and notes from my learning process.

---

##  Project Overview

The goal of this project is to train a convolutional neural network (CNN) on the **CIFAR-10** dataset and:

- Learn the full training pipeline in PyTorch (data loading, augmentation, training loop).
- Experiment with a deeper CNN architecture (BatchNorm, Dropout, LeakyReLU).
- Analyze training curves (loss & accuracy).
- Build a clean, reproducible GitHub project for my ML portfolio.

---

##  Dataset

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Train set:** 50,000 images
- **Test set:** 10,000 images
- **Image size:** 32×32, RGB
- **Number of classes:** 10  
  (`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`)

Data is downloaded automatically by `torchvision.datasets.CIFAR10`.

### Data Augmentation

Used to improve generalization:

- `RandomHorizontalFlip`
- `RandomRotation(20°)`
- `RandomResizedCrop(32, scale = (0.7, 1.0))`
- Normalization to mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)

---

##  Model Architecture

The model is a custom CNN implemented in `src/train.py` with:

- **Convolutional blocks (×5):**
  - Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d
  - Channels: `3 → 64 → 128 → 256 → 512 → 512`
  - Kernel size: 3×3, padding = 1
- **Fully connected head:**
  - Flatten to `512 × 1 × 1`
  - `Linear(512 → 512)` + LeakyReLU + Dropout(0.3)
  - `Linear(512 → 256)` + LeakyReLU
  - `Linear(256 → 10)` output logits
- **Activation:** LeakyReLU
- **Regularization:** BatchNorm + Dropout
- **Loss:** CrossEntropyLoss

This architecture is deeper than a basic CNN but smaller than ResNet-18 – good for learning and experiments.

---

##  Training Configuration (Hyperparameters)

Main training settings:

- **Optimizer:** SGD  
  - Learning rate: `0.01`  
  - Momentum: `0.9`  
  - Weight decay: `5e-4`
- **Scheduler:** `StepLR(step_size = 30, gamma = 0.1)`
- **Batch size:** `32`
- **Full training epochs:** `150` (best model reached ~96% train accuracy)
- **Visualization run (for curves):** `~65` epochs
- **Device:** GPU if available, otherwise CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)
- **Progress bar:** `tqdm` for nice training logs

All of this is implemented in `src/train.py`.

---

##  Results

This repository includes training experiments on CIFAR-10 using the custom CNN described above.

- Training accuracy reaches **~96%** after ~65 epochs.
- Training loss decreases smoothly, showing stable convergence.
- Full training curves are available in the [`results/`](results) folder:

  - [Training loss curve](results/loss_curve.png)
  - [Training accuracy curve](results/accuracy_curve.png)

### Test Evaluation (TODO)

Right now the project focuses on training curves and train accuracy.  
Next step is to add proper evaluation on the **CIFAR-10 test set** and log:

- Test accuracy  
- Confusion matrix  
- Class-wise performance  

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/amit7768/pytorch-cifar10-experiments.git
cd pytorch-cifar10-experiments
