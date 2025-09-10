# OfficialCode

## 📌 Overview
This repository contains a Jupyter Notebook (`OfficialCode256over.ipynb`) implementing a **semantic segmentation task** on **256×256** images. The goal of the notebook is to train and evaluate a segmentation network — based on **U-Net (256)** — to produce masks from input images.

---

## 📂 Notebook Structure

The notebook is organized into the following main sections:

1. **Installation and Imports**
   - Setup of required dependencies (PyTorch, OpenCV, PIL, NumPy, matplotlib).
   - Optional Google Drive mount for dataset access and model saving.

2. **Dataset Loading**
   - Function `load_data(path)` collects and splits images into `train`, `validation`, and `test` sets.
   - It assumes that for each input image there is a corresponding mask (one-to-one pairing).

3. **Custom Dataset**
   - Class `RetinaDataset(torch.utils.data.Dataset)` loads `(image, mask)` pairs, applies transformations (resize to 256×256, normalization, tensor conversion), and optional data augmentation.

4. **Segmentation Model**
   - Implementation of a **U-Net 256** (encoder-decoder with skip connections).
   - Channel-wise normalization (InstanceNorm or BatchNorm where used) and ReLU/LeakyReLU activations.

5. **Losses and Metrics**
   - Primary loss: `BCEWithLogitsLoss` (for raw logits) or `nn.BCE`.
   - Supplementary loss: **Dice Loss** or a combination `BCE + Dice`.
   - Evaluation metrics: **IoU (Intersection over Union)**, **Dice coefficient**, precision, recall, and pixel-wise accuracy.

6. **Testing & Visualization**
   - Mask generation on test images.
   - Overlay visualization between original image and predicted mask.
   - Saving of output images and loss curves.

---

## ⚙️ Requirements
- Python 3.8+
- PyTorch >= 1.9
- torchvision
- OpenCV (optional) / `opencv-python-headless`
- Pillow
- NumPy
- matplotlib

Install dependencies (example):

```bash
pip install torch torchvision opencv-python-headless pillow numpy matplotlib
```

A GPU (CUDA) is recommended for faster execution.

---

## 📂 Dataset Folder Structure
The notebook expects a paired `image-mask` dataset structured as follows:

```
dataset_root/
├── training/
│   ├── images/
│   │   ├── img_0001.png
│   │   ├── img_0002.png
│   │   └── ...
│   └── masks/
│       ├── img_0001_mask.png
│       ├── img_0002_mask.png
│       └── ...
├── validation/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

File names must be paired (e.g., `img_0001.png` ↔ `img_0001_mask.png`). If the dataset is organized differently, adjust the `load_data()` function accordingly.

---

## 📊 Expected Outputs
- Saved model weights (e.g., `unet_epochXX.pth`)
- Training/validation curves for losses and metrics
- Test images with predicted masks and overlays
- Metric reports (IoU/Dice) on the test set

---

## 💡 Suggestions and Possible Improvements
- Balance classes using weighted loss or balanced sampling if masks are heavily imbalanced.
- Add data augmentation (rotations, flips, brightness variations) to improve robustness.
- Replace the `BCE + Dice` combination with **Focal Loss** if many areas are easy to classify.
- Evaluate metrics at different thresholds on the logits (default 0.5) or optimize the threshold selection.
