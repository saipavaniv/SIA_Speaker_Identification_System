import json
import os
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"


st.set_page_config(
    page_title="SIA Speaker Identification",
    layout="wide"
)


st.title("Speaker Identification System")
st.write(
    "This ML application identifies speakers from audio samples using MFCC feature extraction "
    "and classical machine learning models such as SVM, Random Forest and KNN."
)


st.header("Project Pipeline")
st.markdown(
    "**Raw Audio → MFCC Feature Extraction → Model Training → Evaluation → Saved Model → Prediction**"
)


st.header("Evaluation Metrics")

results_file = MODELS_DIR / "evaluation_results.json"

if results_file.exists():
    with open(results_file, "r") as f:
        results = json.load(f)

    df = pd.DataFrame(results).T
    st.dataframe(df, use_container_width=True)

    if "accuracy" in df.columns:
        best_model = df["accuracy"].idxmax()
        best_accuracy = df.loc[best_model, "accuracy"]
        st.success(f"Best Model: {best_model} with accuracy {best_accuracy:.4f}")
else:
    st.warning("evaluation_results.json not found inside models folder.")


st.header("Model Comparison Graph")

comparison_plot = PLOTS_DIR / "model_comparison.png"

if comparison_plot.exists():
    st.image(str(comparison_plot), caption="Model Comparison")
else:
    st.warning("model_comparison.png not found inside plots folder.")


st.header("Confusion Matrices")

cols = st.columns(3)

plot_files = [
    ("KNN", "confusion_matrix_knn.png"),
    ("Random Forest", "confusion_matrix_random_forest.png"),
    ("SVM", "confusion_matrix_svm.png"),
]

for col, (name, filename) in zip(cols, plot_files):
    with col:
        path = PLOTS_DIR / filename
        if path.exists():
            st.image(str(path), caption=f"{name} Confusion Matrix")
        else:
            st.warning(f"{filename} not found.")


st.header("Real-Time Speaker Prediction")

uploaded_file = st.file_uploader(
    "Upload a WAV audio file for speaker prediction",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        model_path = MODELS_DIR / "knn_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"

        if not model_path.exists():
            st.error("knn_model.pkl not found inside models folder.")
        elif not scaler_path.exists():
            st.error("scaler.pkl not found inside models folder.")
        else:
            signal, sr = librosa.load(temp_path, sr=16000)

            mfcc = librosa.feature.mfcc(
                y=signal,
                sr=sr,
                n_mfcc=13
            )

            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)

            features_scaled = scaler.transform(mfcc_mean)

            prediction = model.predict(features_scaled)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities) * 100)

            st.success(f"This audio belongs to Speaker {prediction}")

            if confidence is not None:
                st.info(f"Prediction Confidence: {confidence:.2f}%")

            st.markdown(
                """
                **Prediction Process:**  
                Audio file → MFCC extraction → Feature scaling → KNN model prediction → Speaker output
                """
            )

    except Exception as e:
        st.error("Prediction failed. Please upload a valid WAV audio file.")
        st.exception(e)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


st.header("MLOps Components Used")
st.markdown(
    """
    - **GitHub** for version control  
    - **Docker** for containerization  
    - **GitHub Actions** for CI/CD automation  
    - **Kubernetes** for deployment and orchestration  
    - **Metrics visualization** using Streamlit  
    """
)