# CIFAR-10 Image Classification with CNN — Project Report

**Author:** Skander | Ironhack AI Engineering Bootcamp | W3 D3-D5 Project 1

---

## 1. Approach

The goal was to build a Convolutional Neural Network (CNN) that classifies CIFAR-10
images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse,
ship, truck). CIFAR-10 was chosen over Animal-10 as it was already familiar from
the prior lab, allowing more time for iteration.

The project followed an iterative approach: starting from a simple baseline CNN,
then testing one targeted change at a time (more filters, data augmentation,
dropout, more epochs, a lower learning rate, and transfer learning) to isolate
what actually improved performance.

## 2. Data Preprocessing

- **Dataset:** CIFAR-10 (60,000 images, 32×32 pixels, 10 balanced classes),
  loaded via `tensorflow.keras.datasets.cifar10`.
- **Split:** 45,000 training / 5,000 validation (10% held out from the official
  50,000 training images) / 10,000 test (official, untouched split).
- **Normalization:** pixel values scaled from [0, 255] to [0, 1] to stabilize
  and speed up training.
- **Data Augmentation** (from Model 3 onward): random rotation (±15°), width/height
  shift (±10%), and horizontal flip, applied live during training via
  `ImageDataGenerator`. No vertical flip was used, as it would produce unrealistic
  images (e.g. upside-down vehicles).

## 3. Model Architecture

All custom CNNs followed the same base structure:

```
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
Conv2D(64, 3x3, relu) → MaxPooling2D(2x2)
Conv2D(64 or 128, 3x3, relu)
Flatten()
Dense(64, relu)
Dense(10, softmax)
```

Filter counts increase with depth (32 → 64 → 64/128) while spatial resolution
shrinks after each pooling layer — early layers detect simple features (edges,
color transitions), deeper layers detect more abstract, class-specific patterns.

For transfer learning (Model 7), a frozen, ImageNet-pretrained **VGG16** was used
as a feature extractor (input resized from 32×32 to 96×96), followed by the same
custom Dense head.

## 4. Training Details

- **Optimizer:** Adam (default learning rate 0.001, except Model 6: 0.0005)
- **Loss:** sparse categorical crossentropy
- **Batch size:** 64
- **Epochs:** 10 (Models 1–4, 7), 25 with Early Stopping (Model 5), 5/interrupted (Model 6)
- **Early Stopping:** monitored `val_accuracy`, patience = 5, restored best weights
  (introduced from Model 5 onward)

## 5. Results — All Models

| Model | Architecture | Epochs | Train Acc | Val Acc | Test Acc | Key Takeaway |
|---|---|---|---|---|---|---|
| 1 | 3 Conv (32-64-64), baseline | 10 | 74.0% | 68.7% | 68.0% | Baseline, mild overfitting (gap 5.3%) |
| 2 | 3 Conv (32-64-128) | 10 | 79.0% | 71.1% | 70.6% | More filters helped, but larger gap (7.9%) |
| 3 | + Data Augmentation | 10 | 67.8% | 70.5% | 70.5% | Overfitting eliminated (gap -2.8%) |
| 4 | + Augmentation + Dropout(0.3) | 10 | 66.5% | 70.3% | — | Dropout added no benefit over augmentation alone |
| **5** | **+ Augmentation, 25 epochs + Early Stopping** | **25** | **75.1%** | **77.4%** | **77.1%** | **Best model** — more training time was the strongest lever |
| 6 | + Learning rate 0.0005 | 5 (interrupted) | 59.9% | 64.3% | ~64% | Training repeatedly interrupted (Colab connection issue); not fairly comparable |
| 7 | Transfer Learning (VGG16, frozen) | 10 | 85.3% | 75.7% | 75.8% | Slightly below Model 5, more overfitting (gap 9.6%) |

## 6. Best Model and Why

**Model 5** (custom CNN, 32-64-128 filters, data augmentation, 25 epochs with
early stopping) achieved the best result: **77.4% validation accuracy / 77.1%
test accuracy**, with only a small train/validation gap (~2.3%), indicating good
generalization rather than memorization.

It outperformed transfer learning with VGG16 (Model 7, 75.8%), likely because
VGG16 was pretrained on large, natural photographs (ImageNet), which differ
substantially from CIFAR-10's small, low-resolution images — the upscaling from
32×32 to 96×96 likely introduced artifacts that a from-scratch CNN did not have
to deal with.

## 7. Detailed Evaluation of the Best Model (Model 5)

**Classification report (test set, 10,000 images):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| airplane | 0.76 | 0.84 | 0.80 |
| automobile | 0.85 | 0.90 | 0.87 |
| bird | 0.73 | 0.64 | 0.68 |
| cat | 0.63 | 0.52 | 0.57 |
| deer | 0.74 | 0.74 | 0.74 |
| dog | 0.69 | 0.69 | 0.69 |
| frog | 0.85 | 0.82 | 0.84 |
| horse | 0.76 | 0.83 | 0.80 |
| ship | 0.89 | 0.84 | 0.87 |
| truck | 0.77 | 0.88 | 0.82 |
| **Overall accuracy** | | | **0.77** |

**Confusion matrix insights:**
- Strongest classes: automobile (895/1000), truck (878), ship (842), horse (833),
  frog (825) — clearly separated, little confusion with other classes.
- Weakest class: **cat** (only 517/1000 correct). The dominant confusion is
  **cat ↔ dog** (162 cats predicted as dog, 134 dogs predicted as cat) — a
  reasonable error, since both are furry, four-legged pets with visually
  overlapping features.
- A secondary confusion pair is **automobile ↔ truck** (66 + 68 cases),
  both being vehicles with similar silhouettes.
- Confusion between unrelated classes (e.g. cat↔truck: 32, bird↔automobile: 5)
  is minimal, indicating the model learned meaningful, non-random visual patterns
  rather than memorizing training images.

## 8. Insights from the Experimentation Process

1. **Data augmentation was the single most effective anti-overfitting technique**,
   eliminating the train/validation gap almost entirely on its own.
2. **Dropout, added on top of augmentation, did not provide further improvement**
   — once overfitting was already under control, additional regularization had
   little left to fix.
3. **Training length (epochs) mattered more than expected**: extending from 10 to
   25 epochs (with early stopping as a safeguard) produced the largest single
   accuracy gain (+6.9 percentage points on validation) of any change tested.
4. **Transfer learning is not automatically superior** to a well-tuned custom
   model, especially when the pretrained model's original domain (natural photos)
   differs significantly from the target dataset (small, low-resolution images).
5. **Not all misclassifications are equal**: reviewing the confusion matrix rather
   than accuracy alone revealed that errors clustered around visually similar
   classes (cat/dog, automobile/truck), suggesting the model's mistakes are
   reasonable rather than indicating a fundamentally broken approach.

## 9. Limitations and Notes

- Model 6 (lower learning rate) could not be trained for the full 25 epochs due
  to a recurring Colab connection interruption (cause not fully identified;
  system resources were not exhausted). Its result should be treated as
  preliminary.
- Full precision/recall/F1/confusion matrix analysis was performed for the best
  model (5) only, given time constraints; the same code can be reused for other
  models if a broader comparison is needed.
- Transfer learning was only tested with VGG16; other architectures (e.g.
  Inception, ResNet) might perform differently.

## 10. Project Management Approach

The project was organized using lightweight Agile/Kanban principles adapted for
solo work: a daily self-check (informal stand-up) replaced team stand-ups, an
informal backlog tracked planned model variants, and a simple tracking table
(logged per model: architecture, hyperparameters, accuracy, notes) served as the
project's "single source of truth" instead of a full Kanban board — appropriate
given the project's small scale and single contributor.
