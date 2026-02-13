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

# --- CORE FIX: Smart Column Matching ---
def align_columns(df, model):
    """
    Forces the input DataFrame to match the model's expected columns.
    It handles case sensitivity and creates missing columns with 0s.
    """
    df_processed = df.copy()
    
    # 1. Normalize both input and model columns to lowercase for matching
    df_processed.columns = [c.strip().lower() for c in df_processed.columns]
    
    if hasattr(model, "feature_names_in_"):
        required_cols = list(model.feature_names_in_)
        
        # Create a new dataframe with the exact columns the model needs
        df_final = pd.DataFrame(index=df_processed.index)
        
        for col in required_cols:
            col_lower = col.lower()
            
            # Case 1: Exact match found (ignoring case)
            if col_lower in df_processed.columns:
                df_final[col] = df_processed[col_lower]
            
            # Case 2: Column is "gender_Male" but input is "Gender"
            elif "male" in col_lower and "gender" in df_processed.columns:
                 df_final[col] = df_processed["gender"].apply(lambda x: 1 if str(x).lower() in ['male', '1'] else 0)

            # Case 3: Column is "polyuria_Yes" but input is "Polyuria"
            elif "_yes" in col_lower:
                base_name = col_lower.replace("_yes", "")
                if base_name in df_processed.columns:
                    df_final[col] = df_processed[base_name].apply(lambda x: 1 if str(x).lower() in ['yes', 'positive', '1'] else 0)
                else:
                    df_final[col] = 0 # Missing column -> fill with 0
            
            # Case 4: Completely missing
            else:
                df_final[col] = 0
                
        return df_final
    else:
        return df_processed

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
        
        # Separate Features & Target
        X_raw = df.drop(columns=[target_col])
        y_test = df[target_col]

        X_test = align_columns(X_raw, model)
        
        # Preprocess Target
        mapping = {'Positive': 1, 'Negative': 0, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        if y_test.dtype == 'object':
             y_test = y_test.map(mapping).fillna(0)
        
        # Predict
        try:
            X_array = X_test.values
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
