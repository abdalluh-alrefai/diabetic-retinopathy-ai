# Kaggle Quickstart (copy into a Kaggle Notebook)

> On Kaggle: **New Notebook → Add Data → search "APTOS 2019 Blindness Detection" → Settings → Accelerator: GPU T4 ×2**.
> The dataset mounts at `/kaggle/input/aptos2019-blindness-detection/`.

```python
# 1) Clone your repo (after you push it to GitHub)
!git clone https://github.com/<your-username>/diabetic-retinopathy-ai.git
%cd diabetic-retinopathy-ai
!pip install -q timm albumentations grad-cam pyyaml

# 2) Point config at the Kaggle paths
import yaml
cfg = yaml.safe_load(open("config.yaml"))
cfg["data"]["csv_path"]   = "/kaggle/input/aptos2019-blindness-detection/train.csv"
cfg["data"]["images_dir"] = "/kaggle/input/aptos2019-blindness-detection/train_images"
cfg["data"]["image_ext"]  = ".png"
yaml.safe_dump(cfg, open("config.yaml", "w"))

# 3) Train (saves checkpoints/best.pt)
!python -m src.train --config config.yaml

# 4) Evaluate -> QWK + confusion matrix
!python -m src.evaluate --config config.yaml --ckpt checkpoints/best.pt

# 5) Explain a single image with Grad-CAM
!python -m src.gradcam --ckpt checkpoints/best.pt \
    --image /kaggle/input/aptos2019-blindness-detection/train_images/0a4e1a29ffff.png
```

**Tips for a strong leaderboard score**
- Train 2–3 different backbones (`tf_efficientnet_b4_ns`, `convnext_small.fb_in22k`) and **ensemble** their softmax outputs.
- Add **TTA** (test-time augmentation: average predictions over flips/rotations).
- Use **5-fold stratified CV** and average folds.
- Combine APTOS + the larger 2015 EyePACS dataset for extra training data.
