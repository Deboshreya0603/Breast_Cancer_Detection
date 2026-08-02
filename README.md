# 🧬 PrismOnco | Breast Cancer Prediction

**PrismOnco** is an AI-powered diagnostic web application that predicts the likelihood of breast cancer (benign or malignant) based on user-provided diagnostic inputs. Built with **Streamlit**, it combines a **K-Nearest Neighbors (KNN)** classifier trained on the **Breast Cancer Wisconsin Diagnostic Dataset** with a modern, glassmorphism-inspired interface and interactive model performance visualizations.

---

## 🚀 Features

- **🧠 AI-Powered Diagnosis** — Predicts whether a tumor is benign or malignant using a KNN classification model.
- **🎨 Modern Glassmorphic UI** — Custom-designed landing page and dynamic layout built with hand-crafted CSS.
- **📊 Dataset Insights** — Displays dataset statistics and feature breakdowns for transparency.
- **💬 Symptom-Based Input** — Converts real-world symptom inputs into ML-ready feature vectors dynamically.
- **📈 Interactive Visualizations**
  - Confusion matrix (Plotly)
  - ROC curve and AUC score
  - Classification report (precision, recall, F1-score)
- **🔍 Explainable Predictions** — Users can inspect the generated input features and understand model confidence behind each prediction.

---

## 🧩 Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend / UI** | Streamlit, HTML, CSS (Glassmorphism styling), Plotly |
| **Machine Learning** | Scikit-learn (KNN Classifier, StandardScaler) |
| **Dataset** | Breast Cancer Wisconsin Diagnostic Dataset |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Express, Plotly Graph Objects |

---

## 📸 Preview

*(Add a screenshot or GIF of the app here — e.g. `![PrismOnco Demo](assets/demo.gif)`)*

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/RatnadeepMukherjee/PrismOnco.git
cd PrismOnco

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📂 Project Structure

```
PrismOnco/
├── app.py                 # Main Streamlit application
├── model/                 # Trained KNN model & scaler
├── data/                  # Dataset files
├── assets/                # UI assets (CSS, images)
├── requirements.txt
└── README.md
```

---

## 📊 Model Overview

The classifier is trained on the Breast Cancer Wisconsin Diagnostic Dataset, using **StandardScaler** for feature normalization and a **KNN classifier** for prediction. Performance is evaluated using confusion matrix analysis, ROC-AUC scoring, and a full classification report (precision, recall, F1-score), all rendered interactively within the app.
