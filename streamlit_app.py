import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Set Page Config
st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

st.title("Early Stage Diabetes Risk Prediction")
st.write("**M.Tech (AIML/DSE) - Assignment 2**")

# --- HELPER: PREPROCESSING ---
def preprocess_input(df, model):
    """
    Manually converts raw data to match the One-Hot Encoded format 
    (drop_first=True) used during training.
    """
    df = df.copy()
    # 1. Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # 2. Define Binary Mappings (mimicking drop_first=True)
    binary_features = {
        'gender': 'Male', 'polyuria': 'Yes', 'polydipsia': 'Yes', 
        'sudden_weight_loss': 'Yes', 'weakness': 'Yes', 'polyphagia': 'Yes', 
        'genital_thrush': 'Yes', 'visual_blurring': 'Yes', 'itching': 'Yes', 
        'irritability': 'Yes', 'delayed_healing': 'Yes', 'partial_paresis': 'Yes', 
        'muscle_stiffness': 'Yes', 'alopecia': 'Yes', 'obesity': 'Yes'
    }

    # 3. Apply Encoding
    for col, positive_value in binary_features.items():
        if col in df.columns:
            encoded_col_name = f"{col}_{positive_value}"
            df[encoded_col_name] = df[col].apply(lambda x: 1 if str(x).strip() == positive_value else 0)
            df.drop(col, axis=1, inplace=True)

    # 4. Align columns with the model
    if hasattr(model, "feature_names_in_"):
        try:
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        except Exception as e:
            st.error(f"Error aligning columns: {e}")
            st.stop()
    return df

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("Configuration")

# [Requirement B] Model selection dropdown (1 Mark)
model_dir = "model"
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'scaler' not in f]
    selected_model_file = st.sidebar.selectbox("Select Classification Model", model_files)
else:
    st.error("Model directory not found.")
    st.stop()

# --- MAIN SECTION ---

# [Requirement A] Dataset upload option (CSV) (1 Mark)
st.write("### 1. Upload Test Data")
st.info("Upload a CSV file containing test samples. The app will generate predictions and compare them to the actual labels.")
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None and selected_model_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.write("**Data Preview:**", df.head())
    
    # Select Target
    target_col = st.selectbox("Select Target Column (Actual Labels)", df.columns, index=len(df.columns)-1)
    
    if st.button("Run Prediction"):
        try:
            # Load Assets
            model_path = os.path.join(model_dir, selected_model_file)
            model = joblib.load(model_path)
            
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            
            # Prepare Features & Target
            X_raw = df.drop(columns=[target_col])
            y_test = df[target_col]

            # Preprocess & Scale
            X_test = preprocess_input(X_raw, model)
            if scaler:
                X_test_scaled = scaler.transform(X_test.values) # Pass values only!
            else:
                X_test_scaled = X_test.values

            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Map Target to 0/1 for Metrics
            mapping = {'Positive': 1, 'Negative': 0, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
            if y_test.dtype == 'object':
                 y_test = y_test.map(mapping).fillna(0)
            
            # --- [Requirement C] Display of evaluation metrics (1 Mark) ---
            st.write("### 2. Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col1.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            
            col2.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            col2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            
            col3.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # --- [Requirement D] Confusion matrix or classification report (1 Mark) ---
            st.write("### 3. Detailed Analysis")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)
                
            with c2:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose())
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
