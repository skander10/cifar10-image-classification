# CIFAR-10 Image Classification with CNN
### Presentation — Ironhack AI Engineering Bootcamp, Project 1
**Skander**

---

## Slide 1 — Title
**CIFAR-10 Image Classification with a CNN**
A 3-day Deep Learning project: build, train, evaluate.

---

## Slide 2 — Goal & Dataset
- Build a CNN that classifies images into 10 categories
- Dataset: **CIFAR-10** — 60,000 images, 32×32 pixels, 10 balanced classes
  (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- 45,000 train / 5,000 validation / 10,000 test split

*(Optional: show the sample-images grid from Notebook 1 here)*

---

## Slide 3 — Approach
Iterative, one-change-at-a-time strategy:
1. Start with a simple baseline CNN
2. Test one targeted change per model (filters, augmentation, dropout,
   training length, learning rate, transfer learning)
3. Track every result to isolate what actually works

→ 7 models built and compared over 3 days

---

## Slide 4 — Model Architecture (Base Design)
```
Conv2D(32) → Pool → Conv2D(64) → Pool → Conv2D(64/128)
→ Flatten → Dense(64) → Dense(10, softmax)
```
- Filters increase with depth (32→64→128): simple → abstract features
- Spatial size shrinks after each pooling layer
- Optimizer: Adam | Loss: sparse categorical crossentropy

---

## Slide 5 — All 7 Models — Results

| Model | Key change | Val Acc |
|---|---|---|
| 1 | Baseline | 68.7% |
| 2 | More filters (128) | 71.1% |
| 3 | + Data Augmentation | 70.5% |
| 4 | + Dropout | 70.3% |
| **5** | **+ 25 epochs, Early Stopping** | **77.4%** |
| 6 | Lower learning rate | 64.3%* |
| 7 | Transfer Learning (VGG16) | 75.7% |

*\*Model 6: training interrupted by a technical (Colab) issue*

---

## Slide 6 — What Actually Helped?
- ✅ **Data Augmentation** → eliminated overfitting almost completely
- ✅ **More training time** (25 vs 10 epochs) → biggest single accuracy gain (+6.9%)
- ❌ **Dropout on top of augmentation** → no additional benefit
- ❌ **Transfer Learning (VGG16)** → slightly worse than custom CNN + more overfitting
  (ImageNet's natural photos ≠ CIFAR-10's small images)

**Key lesson:** not every "best practice" technique helps in every situation —
testing one variable at a time revealed which changes actually mattered.

---

## Slide 7 — Best Model: Model 5
**Custom CNN + Data Augmentation + 25 epochs + Early Stopping**
- Validation Accuracy: **77.4%**
- Test Accuracy: **77.1%**
- Train/Val gap: only ~2.3% → good generalization, minimal overfitting

---

## Slide 8 — Evaluation: Confusion Matrix
*(Insert confusion matrix image here)*

- Strongest classes: automobile (90% recall), ship, truck, frog, horse
- Weakest class: **cat** (52% recall)
- Main confusion: **cat ↔ dog** (similar furry, four-legged pets)
- Secondary confusion: **automobile ↔ truck** (similar vehicle shapes)
- Almost no confusion between unrelated classes → model learned real,
  meaningful visual patterns, not random guesses

---

## Slide 9 — Example Errors
*(Insert the 10 misclassified-image grid here: True vs Predicted labels)*

Most errors are "reasonable" mistakes (e.g. cat predicted as dog) rather
than nonsensical ones (e.g. cat predicted as truck) — a sign of a
well-generalizing model.

---

## Slide 10 — Project Organization
- Solo work, adapted from the team Kanban/Agile framework introduced at kick-off
- Daily self-check instead of team stand-up
- Simple results table (architecture, hyperparameters, accuracy, notes)
  as single source of truth — logged after every model
- All decisions and progress documented continuously (Notion)

---

## Slide 11 — Challenges & How They Were Solved
- **Colab session drops** → lost in-memory data/models → solved by saving
  all preprocessed data & trained models to Google Drive
- **Repeated training interruption** (Model 6) → solved with `ModelCheckpoint`
  to save progress automatically
- **Overfitting** (Models 1-2) → solved via data augmentation

---

## Slide 12 — Conclusion & Next Steps
- Best model: **Model 5, 77.1% test accuracy**
- Full results, code, and analysis: see GitHub repo + REPORT.md
- Possible future improvements: more epochs, additional augmentation types,
  fine-tuning (unfreezing) upper VGG16 layers, ensembling models 5 + 7

**Thank you! Questions?**
