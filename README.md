# CIFAR-10 Image Classification with CNN

3-day project — Ironhack AI Engineering Bootcamp (W3 D3-D5).

## Goal
Build, train, and evaluate a Convolutional Neural Network (CNN) 
that classifies images from the CIFAR-10 dataset into 10 categories.

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32x32 
color images, 10 classes, 6,000 images per class.

## Project Structure
notebooks/   → 01_baseline_model.ipynb, 02_augmentation_and_more_models.ipynb
models/      → saved (pickled) trained models
REPORT.md    → full project report with results and analysis
requirements.txt → dependencies

## Approach
Iterative model building: starting from a simple baseline CNN, then testing
one targeted change at a time — more filters, data augmentation, dropout,
extended training with early stopping, learning rate tuning, and transfer
learning (VGG16) — to isolate what actually improved performance.

## Results

| Model | Key change | Val Accuracy | Test Accuracy |

| 1 | Baseline (3 conv layers) | 68.7% | 68.0% |
| 2 | More filters (32-64-128) | 71.1% | 70.6% |
| 3 | + Data Augmentation | 70.5% | 70.5% |
| 4 | + Dropout | 70.3% | — |
| **5** | **+ 25 epochs, Early Stopping** | **77.4%** | **77.1%** |
| 6 | Lower learning rate | 64.3% | ~64% |
| 7 | Transfer Learning (VGG16) | 75.7% | 75.8% |

**Best model: Model 5** — custom CNN with data augmentation and extended
training (25 epochs, early stopping). See [REPORT.md](./REPORT.md) for full
details, confusion matrix analysis, and insights.

## How to Run
```bash
pip install -r requirements.txt
```
Open the notebooks in Google Colab (recommended, GPU support) or Jupyter,
and run cells in order.

## Author
Skander — Ironhack AI Engineering Bootcamp
