<h1 align="center">👁️ RetinaScan AI — Diabetic Retinopathy Detection</h1>

<p align="center">
  <b>Explainable deep learning for grading diabetic retinopathy from retinal fundus images.</b><br>
  End-to-end: preprocessing → training → evaluation → Grad-CAM explainability → web app & REST API.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg">
  <img src="https://img.shields.io/badge/timm-EfficientNet%20%7C%20ConvNeXt-success.svg">
  <img src="https://img.shields.io/badge/explainability-Grad--CAM-orange.svg">
  <img src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

---

## 🩺 Problem

Diabetic retinopathy (DR) is a leading cause of preventable blindness. Diagnosis
requires a specialist to grade retinal scans on a 0–4 scale — slow, costly, and
scarce in many regions. **RetinaScan AI** automates that grading and, crucially,
**shows *why*** it made each decision via Grad-CAM heatmaps, making it
trustworthy enough for clinical conversation.

| Grade | Meaning |
|:-----:|---------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## ✨ Highlights

- **State-of-the-art backbones** via [`timm`](https://github.com/huggingface/pytorch-image-models) — EfficientNet-B4 / ConvNeXt, ImageNet-pretrained.
- **Ben Graham fundus preprocessing** (border crop + colour normalisation) — the technique from the original winning DR solution.
- **Class-imbalance handling** with weighted loss (No-DR dominates the data).
- **Official metric**: Quadratic Weighted Kappa (QWK), plus accuracy & confusion matrix.
- **Explainability**: Grad-CAM heatmaps highlight the retinal regions driving each prediction.
- **Deployment**: a Streamlit web app **and** a FastAPI REST endpoint.
- **Reproducible**: single `config.yaml`, fixed seeds, stratified split, mixed-precision training.

## 🖼️ Demo

> _Add screenshots after your first run:_
> `assets/demo_app.png` (Streamlit) and `assets/gradcam_example.png` (heatmap).

## 📊 Results

Evaluated on a held-out **test set** (366 images the model never saw). Metric: Quadratic Weighted Kappa (QWK) — the official APTOS metric.

| Model | Test QWK | Accuracy | Mild recall | Severe recall | Macro recall |
|-------|:--------:|:--------:|:-----------:|:-------------:|:------------:|
| EfficientNet-B0 (256px, baseline) | 0.883 | 0.81 | 0.43 | 0.35 | 0.61 |
| **EfficientNet-B3 + Focal Loss + balanced sampling + TTA (300px)** | 0.872 | 0.79 | **0.63** | **0.53** | **0.67** |

**Why the final model wins despite a marginally lower QWK:** balanced sampling and
Focal Loss were used deliberately to raise recall on the clinically critical rare
grades. Severe-DR recall improved from 0.35 → 0.53 and Mild from 0.43 → 0.63, at a
negligible (~0.01) cost in overall agreement. In a screening context, missing a
severe case is far costlier than a marginal QWK drop — so the balanced model is the
clinically sounder choice.

## 🗂️ Project structure

```
diabetic-retinopathy-ai/
├── config.yaml              # all hy-parameters & paths in one place
├── requirements.txt
├── src/
│   ├── preprocess.py        # Ben Graham fundus preprocessing
│   ├── dataset.py           # PyTorch Dataset + Albumentations
│   ├── model.py             # timm model factory
│   ├── train.py             # training loop (AMP, QWK, early stopping)
│   ├── evaluate.py          # QWK, report, confusion matrix
│   └── gradcam.py           # Grad-CAM explainability
├── app/
│   ├── app.py               # Streamlit demo
│   └── api.py               # FastAPI inference service
└── notebooks/
    └── 01_kaggle_quickstart.md
```

## 🚀 Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. get the data (Kaggle: APTOS 2019 Blindness Detection)
#    place train.csv + train_images/ under ./data and edit config.yaml

# 3. train
python -m src.train --config config.yaml

# 4. evaluate
python -m src.evaluate --config config.yaml --ckpt checkpoints/best.pt

# 5. explain one image
python -m src.gradcam --ckpt checkpoints/best.pt --image data/train_images/xxxx.png

# 6. run the app  /  the API
streamlit run app/app.py
uvicorn app.api:app --port 8000
```

No local GPU? See [`notebooks/01_kaggle_quickstart.md`](notebooks/01_kaggle_quickstart.md) to train free on Kaggle/Colab.

## 📚 Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
— 3,662 labelled fundus images from Aravind Eye Hospital (India). Optionally
combine with the larger [EyePACS 2015](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
set (~88k images) for stronger generalisation.

## 🧠 Tech stack

PyTorch · timm · Albumentations · OpenCV · scikit-learn · pytorch-grad-cam · Streamlit · FastAPI

## ⚠️ Disclaimer

Research and educational project only. **Not** a certified medical device and
not a substitute for professional diagnosis.

## 📄 License

MIT © Abdallah Alrefae
