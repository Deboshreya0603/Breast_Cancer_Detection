# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.datasets import load_breast_cancer
# from landing import show_landing
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (accuracy_score, confusion_matrix,
#                              precision_score, recall_score,
#                              roc_curve, auc, precision_recall_curve)
# # ======================
# # LANDING PAGE 
# # ======================
# show_landing()

# # ======================
# # Streamlit Page Config
# # ======================
# st.set_page_config(
#     page_title="BreastScan AI | KNN Diagnosis",
#     page_icon="🧬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =================
# # Custom CSS Styling
# # =================

# def inject_custom_css():
#     st.markdown("""
#     <style>
#     /* Base styles for the app */
#     html, body, .stApp {
#         font-family: 'Poppins', sans-serif;
#         background: linear-gradient(to bottom right, #f8fafc, #e0f2fe);
#         color: #1e293b;
#     }

#     h1, h2, h3, h4 {
#         font-weight: 700;
#         letter-spacing: -0.5px;
#     }

#     /* Glassmorphic Card Styles */
#     .card, .prediction-card {
#         background: rgba(70, 130, 180, 0.35);
#         border-radius: 16px;
#         padding: 2rem;
#         backdrop-filter: blur(15px);
#         -webkit-backdrop-filter: blur(15px);
#         border: 1px solid rgba(255, 255, 255, 0.25);
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
#         transition: transform 0.3s ease;
#         margin-bottom: 1.5rem;
#     }

#     .prediction-card:hover {
#         transform: scale(1.02);
#     }

#     /* Metrics and badges */
#     .metric-value {
#         font-size: 1.5rem;
#         font-weight: 700;
#         color: #0f172a;
#     }

#     .badge-success {
#         background-color: #dcfce7;
#         color: #16a34a;
#         padding: 0.4em 0.9em;
#         border-radius: 999px;
#         font-weight: 600;
#     }

#     .badge-danger {
#         background-color: #fee2e2;
#         color: #dc2626;
#         padding: 0.4em 0.9em;
#         border-radius: 999px;
#         font-weight: 600;
#     }

#     /* Stylish header box */
#     .glass-header {
#         background: linear-gradient(135deg, #4f46e5, #06b6d4);
#         padding: 2rem;
#         color: white;
#         text-align: center;
#         border-radius: 20px;
#         box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
#         margin-bottom: 2rem;
#     }

#     /* Plot container */
#     .plot-container {
#         max-width: 600px;
#         margin: auto;
#         border-radius: 16px;
#         overflow: hidden;
#         background: white;
#         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
#     }

#     /* Animated gradient title with glow */
#     .app-title {
#         font-family: 'Poppins', sans-serif;
#         font-size: 3rem;
#         background: linear-gradient(90deg, #6366f1, #ec4899, #14b8a6);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         animation: pulseText 3s ease-in-out infinite;
#         text-align: center;
#         margin-bottom: 0;
#         letter-spacing: -1px;
#     }

#     .subtitle {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1.25rem;
#         color: #64748b;
#         text-align: center;
#         margin-top: 0.5rem;
#         animation: fadeIn 2s ease forwards;
#         opacity: 0;
#     }

#     @keyframes pulseText {
#         0%, 100% { text-shadow: 0 0 8px rgba(236, 72, 153, 0.4); }
#         50% { text-shadow: 0 0 16px rgba(20, 184, 166, 0.6); }
#     }

#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(10px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     @media (max-width: 768px) {
#         .app-title {
#             font-size: 2.25rem;
#         }
#         .subtitle {
#             font-size: 1rem;
#         }
#     }
#     </style>
    
#     <!-- Google Fonts -->
#     <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
#     """, unsafe_allow_html=True)

# inject_custom_css()

# # =================
# # Load Breast Cancer Data
# # =================

# @st.cache_data
# def load_data():
#     data = load_breast_cancer()
#     X = pd.DataFrame(data.data, columns=data.feature_names)
#     y = pd.Series(data.target, name='target')

#     # Feature units
#     feature_units = {
#         "mean radius": "(mm)",
#         "mean texture": "(units)",
#         "mean perimeter": "(mm)",
#         "mean area": "(mm²)",
#         "mean smoothness": "(relative)",
#         "mean compactness": "(relative)",
#         "mean concavity": "(relative)",
#         "mean concave points": "(relative)",
#         "mean symmetry": "(relative)",
#         "mean fractal dimension": "(relative)"
#     }

#     # Feature descriptions
#     feature_descriptions = {
#         "mean radius": "Average distance from center to points on perimeter",
#         "mean texture": "Standard deviation of gray-scale values",
#         "mean perimeter": "Tumor perimeter measurement",
#         "mean area": "Area measurement of tumors",
#         "mean smoothness": "Local variation in radius lengths",
#         "mean compactness": "Perimeter² / area - 1.0",
#         "mean concavity": "Severity of concave portions of contour",
#         "mean concave points": "Number of concave portions of contour",
#         "mean symmetry": "Tumor symmetry measurement",
#         "mean fractal dimension": "Coastline approximation - 1.0"
#     }

#     # Feature explanations with healthy ranges
#     feature_explanations = {
#         "mean radius": {
#             "explanation": "How wide the tumor is from its center to the edge.",
#             "healthy_range": "≤ 14.0"
#         },
#         "mean texture": {
#             "explanation": "How rough or grainy the tumor appears.",
#             "healthy_range": "≤ 20.0"
#         },
#         "mean perimeter": {
#             "explanation": "The total distance around the tumor.",
#             "healthy_range": "≤ 90.0"
#         },
#         "mean area": {
#             "explanation": "The full surface area of the tumor.",
#             "healthy_range": "≤ 600"
#         },
#         "mean smoothness": {
#             "explanation": "How smooth or uneven the edge of the tumor is.",
#             "healthy_range": "≤ 0.10"
#         },
#         "mean compactness": {
#             "explanation": "How tightly packed the tumor appears.",
#             "healthy_range": "≤ 0.15"
#         },
#         "mean concavity": {
#             "explanation": "How deeply the tumor's edges are curved inward.",
#             "healthy_range": "≤ 0.15"
#         },
#         "mean concave points": {
#             "explanation": "How many dips/curves appear on the edge.",
#             "healthy_range": "≤ 0.07"
#         },
#         "mean symmetry": {
#             "explanation": "How symmetrical the tumor is on both sides.",
#             "healthy_range": "≤ 0.20"
#         },
#         "mean fractal dimension": {
#             "explanation": "How detailed or complex the tumor edge is.",
#             "healthy_range": "≤ 0.07"
#         }
#     }

#     return X, y, data, feature_units, feature_descriptions, feature_explanations


# # Load the data
# X, y, data, feature_units, feature_descriptions, feature_explanations = load_data()


# # =================
# # Sidebar Info
# # =================
# with st.sidebar:
#     st.markdown("## 📊 Dataset Overview")
#     st.success("You're working with the **Breast Cancer Wisconsin** dataset.")

#     st.markdown("""
#     - **Total Samples:** 569  
#     - **Features per Sample:** 30  
#     - **Classification Labels:**  
#         - 🟢 Benign (Non-cancerous)  
#         - 🔴 Malignant (Cancerous)
#     """)

#     st.markdown("---")
#     st.info("Explore the dataset and model results in the main panel.")

#     st.markdown("### 🧬 Tumor Feature Guide")
#     with st.expander("What do these features mean?"):
#         for feature, simple_desc in feature_explanations.items():
#             st.markdown(f"🔹 **{feature}**: {simple_desc}")

# # =================
# # Model Training
# # =================
# k = 5  
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# model = KNeighborsClassifier(n_neighbors=k)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# performance = {
#     'accuracy': accuracy_score(y_test, y_pred),
#     'precision': precision_score(y_test, y_pred),
#     'recall': recall_score(y_test, y_pred),
#     'cm': confusion_matrix(y_test, y_pred),
#     'fpr': roc_curve(y_test, y_proba)[0],
#     'tpr': roc_curve(y_test, y_proba)[1],
#     'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1]),
#     'precision_recall': precision_recall_curve(y_test, y_proba),
#     'auprc': auc(precision_recall_curve(y_test, y_proba)[1], precision_recall_curve(y_test, y_proba)[0])
# }

# # =================
# # Header UI
# # =================
# st.markdown("""
# <div class="glass-header">
#     <h1 class="app-title">PrismOnco</h1>
#     <p class="subtitle">AI-Powered Breast Cancer Diagnosis</p>
# </div>
# """, unsafe_allow_html=True)

# # =================
# # User Input
# # =================
# input_data = {feature: float(X[feature].mean()) for feature in X.columns}
# cols = st.columns(2)
# features_split = np.array_split(X.columns[:10], 2)

# for i, feature_group in enumerate(features_split):
#     with cols[i]:
#         for feature in feature_group:
#             st.markdown(f"**{feature}** {feature_units.get(feature, '')}")
#             st.caption(feature_descriptions.get(feature, ''))
#             input_data[feature] = st.slider(
#                 label=feature,
#                 min_value=float(X[feature].min()),
#                 max_value=float(X[feature].max()),
#                 value=float(X[feature].mean()),
#                 step=0.01,
#                 label_visibility="collapsed",
#                 key=feature
#             )

# input_df = pd.DataFrame([input_data])[X.columns]

# # =================
# # Run Diagnosis Button with Session State
# # =================
# if "predict_clicked" not in st.session_state:
#     st.session_state.predict_clicked = False

# if st.button("🧠 Run Diagnosis"):
#     st.session_state.predict_clicked = True

# # =================
# # Show Results if Button Clicked
# # =================
# if st.session_state.predict_clicked:
#     try:
#         input_scaled = scaler.transform(input_df)
#         prediction = model.predict(input_scaled)[0]
#         proba = model.predict_proba(input_scaled)[0]

#         st.markdown("### 🔍 Prediction Result")
#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("#### Diagnosis")
#             st.markdown(f"""
#             <div class="prediction-card" style="border-left: 4px solid {'#16a34a' if prediction == 1 else '#dc2626'};">
#                 <span class="{'badge-success' if prediction == 1 else 'badge-danger'}">
#                     {data.target_names[prediction].upper()}
#                 </span>
#                 <p style="margin-top: 0.5rem;">Confidence: <strong>{max(proba)*100:.1f}%</strong></p>
#             </div>
#             """, unsafe_allow_html=True)

#         with col2:
#             st.markdown("#### Probability Breakdown")
#             st.markdown(f"""
#             <div style="text-align: center;">
#                 <p style="color: #16a34a;">Benign: <span class="metric-value">{proba[1]*100:.1f}%</span></p>
#                 <p style="color: #dc2626;">Malignant: <span class="metric-value">{proba[0]*100:.1f}%</span></p>
#             </div>
#             """, unsafe_allow_html=True)

#         # Show performance tabs
#         st.markdown("## 📊 Model Performance")
#         tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix", "ROC Curve"])

#         with tab1:
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Accuracy", f"{performance['accuracy']*100:.2f}%")
#             col2.metric("Precision", f"{performance['precision']*100:.2f}%")
#             col3.metric("Recall", f"{performance['recall']*100:.2f}%")

#         with tab2:
#             fig_cm = px.imshow(
#                 performance['cm'],
#                 text_auto=True,
#                 labels=dict(x="Predicted", y="Actual"),
#                 x=['Malignant', 'Benign'],
#                 y=['Malignant', 'Benign'],
#                 color_continuous_scale='Blues'
#             )
#             fig_cm.update_layout(title="Confusion Matrix", width=500)
#             st.plotly_chart(fig_cm, use_container_width=True)

#         with tab3:
#             fig_roc = go.Figure()
#             fig_roc.add_trace(go.Scatter(
#                 x=performance['fpr'], y=performance['tpr'],
#                 name=f'ROC Curve (AUC = {performance["roc_auc"]:.2f})',
#                 mode='lines', line=dict(color='royalblue', width=3)
#             ))
#             fig_roc.add_shape(
#                 type='line', line=dict(dash='dash', color='gray'),
#                 x0=0, x1=1, y0=0, y1=1
#             )
#             fig_roc.update_layout(
#                 title="Receiver Operating Characteristic Curve",
#                 xaxis_title="False Positive Rate",
#                 yaxis_title="True Positive Rate",
#                 height=500
#             )
#             st.plotly_chart(fig_roc, use_container_width=True)

#     except Exception as e:
#         st.error(f"Prediction Error: {e}")
# else:
#     st.info("⬅️ Adjust the input sliders and click **Run Diagnosis** to see prediction and model performance.")

# # =================
# # Footer
# # =================
# st.markdown("""
# <hr>
# <div style="text-align: center; font-size: 0.85rem; color: #64748b;">
#     BreastScan AI • KNN-Based Diagnostic Tool • For research use only
# </div>
# """, unsafe_allow_html=True)


# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# Optional: Import custom landing screen if available
try:
    from landing import show_landing
    show_landing()
except ImportError:
    pass  # Skip if landing.py not found

# ----------------- Page Config -----------------
st.set_page_config(page_title="PrismOnco | Breast Cancer Prediction", page_icon="🧬", layout="wide")

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@600;800&display=swap');
        .glass-header {
            margin: auto;
            padding: 2rem 3rem;
            max-width: 800px;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.25);
            backdrop-filter: blur(12px);
            text-align: center;
            animation: fadeIn 1s ease-out;
        }
        .glass-header:hover {
            transform: scale(1.02);
            background: linear-gradient(135deg, rgba(255,255,255,0.3), rgba(255,255,255,0.1));
        }
        .app-title {
            font-family: 'Raleway', sans-serif;
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #8e44ad, #e91e63, #00c9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .subtitle {
            font-family: 'Raleway', sans-serif;
            font-size: 1.4rem;
            background: linear-gradient(90deg, #ffc0cb, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 0.5rem;
        }
        .symptom-section {
            background-color: #f9f9fc;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>

    <div class="glass-header">
        <h1 class="app-title">PrismOnco</h1>
        <p class="subtitle">AI-Powered Breast Cancer Diagnosis</p>
    </div>
""", unsafe_allow_html=True)

# ----------------- Load Dataset -----------------
data = load_breast_cancer()
X_raw = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=data.feature_names)

# ----------------- Train Model -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("📊 Dataset Info")
    st.success("Using Breast Cancer Wisconsin Diagnostic dataset.")
    st.markdown(f"""
    - **Total Samples:** {X.shape[0]}
    - **Features:** {X.shape[1]}
    - **Classes:**  
        - 🟢 Benign  
        - 🔴 Malignant
    """)
    if st.checkbox("Show Feature Descriptions"):
        st.markdown("""
        - **Lump in Breast**: Can indicate tumor presence  
        - **Pain Level**: May relate to inflammation  
        - **Nipple Discharge**: Bloody discharge may signal malignancy  
        - **Skin Changes**: Dimpling/redness may indicate issues  
        - **Swelling**: Suggests lymph involvement  
        - **Change in Breast Shape**: Could signal asymmetry  
        - **Enlarged Lymph Nodes**: Sign of potential spread  
        - **Lump Texture**: Irregular lumps raise concern  
        """)

# ----------------- Symptoms Input -----------------
st.markdown("<div class='symptom-section'>", unsafe_allow_html=True)
st.header("📝 Enter Symptoms")

lump = st.selectbox("Lump in breast", ["No", "Yes"])
pain = st.selectbox("Pain level", ["None", "Mild", "Moderate", "Severe"])
discharge = st.selectbox("Nipple discharge", ["None", "Clear", "Bloody"])
skin_changes = st.selectbox("Skin changes", ["None", "Mild", "Severe"])
swelling = st.selectbox("Swelling", ["None", "Mild", "Severe"])
shape_change = st.selectbox("Change in breast shape", ["No", "Slight", "Yes"])
lymph_nodes = st.selectbox("Enlarged lymph nodes", ["No", "Yes"])
lump_texture = st.selectbox("Lump texture", ["Soft", "Hard", "Irregular"])

# ----------------- Symptom Mapping -----------------
def map_symptoms_to_features():
    input_array = data.data.mean(axis=0).copy()
    feature_names = data.feature_names

    def mod(name, delta):
        idx = list(feature_names).index(name)
        input_array[idx] += delta

    if lump == "Yes": mod("mean radius", 5); mod("mean area", 500); mod("mean perimeter", 30)
    if pain == "Moderate": mod("mean texture", 5)
    elif pain == "Severe": mod("mean texture", 10)
    if discharge == "Bloody": mod("mean symmetry", 0.1)
    if skin_changes == "Mild": mod("mean smoothness", -0.02); mod("mean concavity", 0.05)
    elif skin_changes == "Severe": mod("mean smoothness", -0.04); mod("mean concavity", 0.1)
    if swelling == "Mild": mod("mean area", 200)
    elif swelling == "Severe": mod("mean area", 400)
    if shape_change == "Slight": mod("mean symmetry", 0.05)
    elif shape_change == "Yes": mod("mean symmetry", 0.1)
    if lymph_nodes == "Yes": mod("mean compactness", 0.05)
    if lump_texture == "Hard": mod("mean concave points", 0.02)
    elif lump_texture == "Irregular": mod("mean concave points", 0.04); mod("mean fractal dimension", 0.02)

    df_unscaled = pd.DataFrame([input_array], columns=feature_names)
    return pd.DataFrame(scaler.transform(df_unscaled), columns=feature_names)

# ----------------- Predict Button -----------------
if st.button("🔍 Predict Diagnosis"):
    input_df = map_symptoms_to_features()
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    st.subheader("📄 Diagnosis Result")
    if prediction == 1:
        st.error(f"🔴 Malignant (Cancerous) — Confidence: {pred_proba:.2%}")
    else:
        st.success(f"🟢 Benign (Non-Cancerous) — Confidence: {(1 - pred_proba):.2%}")

    with st.expander("🔬 View Input Features"):
        st.dataframe(input_df)

    with st.expander("📊 Model Performance"):
        st.write("### Evaluation Metrics")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"✅ Accuracy: {acc:.2%}")

        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
        ).transpose()
        st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig_cm)

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure([
            go.Scatter(x=fpr, y=tpr, name="ROC Curve", mode="lines"),
            go.Scatter(x=[0, 1], y=[0, 1], name="Baseline", mode="lines")
        ])
        fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc)
