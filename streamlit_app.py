import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Set Page Config
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Model Deployment")
st.write("M.Tech (AIML) - Assignment 2")

def preprocess_input(df, model):
    """
    Manually converts raw data to match the One-Hot Encoded format 
    (drop_first=True) used during training.
    """
    df = df.copy()
    
    # 1. Standardize column names to lowercase for matching
    df.columns = [c.lower() for c in df.columns]

    # 2. Define Binary Mappings (mimicking drop_first=True)
    # Since drop_first=True was used:
    # 'Gender' (Female/Male) -> Becomes 'gender_Male' (1=Male, 0=Female)
    # 'Polyuria' (No/Yes) -> Becomes 'polyuria_Yes' (1=Yes, 0=No)
    binary_features = {
        'gender': 'Male',
        'polyuria': 'Yes',
        'polydipsia': 'Yes',
        'sudden_weight_loss': 'Yes',
        'weakness': 'Yes',
        'polyphagia': 'Yes',
        'genital_thrush': 'Yes',
        'visual_blurring': 'Yes',
        'itching': 'Yes',
        'irritability': 'Yes',
        'delayed_healing': 'Yes',
        'partial_paresis': 'Yes',
        'muscle_stiffness': 'Yes',
        'alopecia': 'Yes',
        'obesity': 'Yes'
    }

    # 3. Apply Encoding
    for col, positive_value in binary_features.items():
        # Check if the raw column exists (e.g., 'polyuria')
        if col in df.columns:
            # Create the specific column name the model expects (e.g., 'polyuria_Yes')
            encoded_col_name = f"{col}_{positive_value}"
            
            # Map values: positive_value becomes 1, everything else 0
            df[encoded_col_name] = df[col].apply(lambda x: 1 if str(x).strip() == positive_value else 0)
            
            # Drop the original raw column
            df.drop(col, axis=1, inplace=True)

    # 4. Align columns with the model
    # This handles cases where the uploaded CSV is ALREADY encoded (like test_data.csv)
    # or if the columns are in a different order.
    if hasattr(model, "feature_names_in_"):
        try:
            # Reindex forces the dataframe columns to match the model's training columns exactly
            # fill_value=0 ensures any missing columns are filled with 0s
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        except Exception as e:
            st.error(f"Error aligning columns: {e}")
            st.stop()
            
    return df

# 1. Model Selection
model_dir = "model"
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    selected_model_file = st.selectbox("Select a Classification Model", model_files)
else:
    st.error("Model directory not found.")
    st.stop()

# 2. Dataset Upload
st.write("### Upload Test Data (CSV)")
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None and selected_model_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    
    # Auto-select target (usually the last column)
    target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    
    if st.button("Run Prediction"):
        model_path = os.path.join(model_dir, selected_model_file)
        model = joblib.load(model_path)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Separate Features & Target
        X_raw = df.drop(columns=[target_col])
        y_test = df[target_col]

        X_test = preprocess_input(X_raw, model)
        X_test_scaled = scaler.transform(X_test.values)
        
        # Preprocess Target
        mapping = {'Positive': 1, 'Negative': 0, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        if y_test.dtype == 'object':
             y_test = y_test.map(mapping).fillna(0)
        
        # Predict
        try:
            X_array = X_test_scaled.values
            y_pred = model.predict(X_array)
            
            # Metrics
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            col1.metric("Precision", round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            col2.metric("Recall", round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            col2.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            col3.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred), 4))
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            
        except Exception as e:
            st.error(f"Error: {e}")
