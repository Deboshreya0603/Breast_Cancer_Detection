🧬 PrismOnco | Breast Cancer Prediction
PrismOnco is an AI-powered diagnostic web app built with Streamlit that predicts the likelihood of breast cancer (benign or malignant) based on user-provided symptoms.
It leverages the Breast Cancer Wisconsin Diagnostic dataset, applies data preprocessing, and uses a K-Nearest Neighbors (KNN) classifier for classification.
The app also includes beautiful glassmorphism-inspired UI design and interactive model performance visualizations.

🚀 Features
🧠 AI-Powered Diagnosis: Predicts if the tumor is benign or malignant using KNN classification.
🎨 Modern Glassmorphic UI: A sleek landing page and dynamic layout designed with custom CSS.
📊 Dataset Insights: Displays dataset statistics and feature details.
💬 Symptom-Based Input: Converts real-world symptoms into ML model features dynamically.
📈 Interactive Visualizations:
Confusion matrix using Plotly
ROC curve and AUC score visualization
Classification report table with precision, recall, and F1-score
🔍 Explainable Predictions: Users can view generated input features and understand model confidence.

🧩 Tech Stack
Frontend : UI	Streamlit, HTML, CSS (Glassmorphism styling), Plotly
Machine Learning : Scikit-learn (KNN Classifier, StandardScaler)
Dataset	: Breast Cancer Wisconsin Diagnostic dataset
Data Processing : 	Pandas, NumPy
Visualization	: Plotly Express, Plotly Graph Objects
