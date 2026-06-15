# LinkedIn Post + CV Bullet

> Fill in the bracketed numbers `[ ]` with your real results after training.

---

## 🔵 LinkedIn Post (English)

🚀 Excited to share my latest deep learning project: **RetinaScan AI** — an explainable system that grades **Diabetic Retinopathy** from retinal fundus images on a 0–4 severity scale.

Diabetic retinopathy is a leading cause of preventable blindness, and early detection saves sight. I built a full end-to-end pipeline:

🔬 Ben Graham fundus preprocessing + heavy augmentation
🧠 EfficientNet-B4 / ConvNeXt (transfer learning via timm)
⚖️ Class-imbalance handling with weighted loss
📊 Quadratic Weighted Kappa of **[0.9X]** on validation
🔍 Grad-CAM heatmaps so the model explains *why* — not a black box
🌐 Deployed as a Streamlit app + FastAPI REST service

The explainability piece matters most to me: in healthcare, a prediction you can't interpret isn't trustworthy. Grad-CAM lets the model point to the exact retinal lesions behind each decision.

Code + write-up 👉 [GitHub link]
Live demo 👉 [Streamlit link]

Feedback welcome! 🙏

#DeepLearning #ComputerVision #MachineLearning #AI #HealthcareAI #PyTorch #MedicalImaging

---

## 🟢 Shorter / punchier alternative

Just shipped **RetinaScan AI** 👁️ — an explainable deep learning model that grades diabetic retinopathy from retinal scans (0–4) and shows *where* it's looking via Grad-CAM.

EfficientNet + ConvNeXt ensemble · QWK [0.9X] · Streamlit demo · FastAPI.

In medical AI, interpretability beats a black box. 🔍

Code 👉 [link] | Demo 👉 [link]

#AI #ComputerVision #DeepLearning #HealthcareAI #PyTorch

---

## 📄 CV / Résumé bullet (English)

**RetinaScan AI — Explainable Diabetic Retinopathy Detection** · _Python, PyTorch, timm_
- Built an end-to-end deep learning pipeline classifying retinal fundus images into 5 DR severity grades, achieving **[0.9X] Quadratic Weighted Kappa** on the APTOS 2019 dataset.
- Engineered Ben Graham fundus preprocessing and a weighted-loss strategy to handle severe class imbalance; ensembled EfficientNet-B4 and ConvNeXt with test-time augmentation.
- Integrated **Grad-CAM explainability** and deployed the model as a Streamlit web app and a FastAPI REST service.

## 📄 نسخة عربية للسيرة الذاتية

**RetinaScan AI — كشف اعتلال الشبكية السكري القابل للتفسير** · _Python, PyTorch_
- بناء نظام تعلّم عميق متكامل يصنّف صور قاع العين إلى 5 درجات لشدة المرض، بنتيجة **QWK = [0.9X]** على بيانات APTOS 2019.
- معالجة متقدّمة للصور (Ben Graham) ومعالجة عدم توازن البيانات، مع دمج نموذجين (EfficientNet + ConvNeXt) وTTA.
- دمج تقنية **Grad-CAM** لتفسير القرارات، ونشر النموذج كتطبيق ويب (Streamlit) وخدمة REST API (FastAPI).
