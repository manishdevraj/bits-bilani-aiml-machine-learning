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

# Preprocessing & Column Alignment ---
def preprocess_input(df, model):
    """
    Transforms input data to match the model's training features exactly.
    Handles both raw text inputs (e.g., 'Gender') and already encoded inputs.
    """
    df_processed = df.copy()

    # 1. Standardize Column Names to lowercase to avoid case-sensitivity issues
    df_processed.columns = [c.lower() for c in df_processed.columns]

    # 2. Manual One-Hot Encoding (Raw Text -> Encoded Columns)
    
    # Handle Gender: Map 'Male' -> 1 (if column exists)
    if 'gender' in df_processed.columns:
        # Check for various forms of "Male"
        df_processed['gender_Male'] = df_processed['gender'].apply(lambda x: 1 if str(x).strip().lower() in ['male', '1'] else 0)
        df_processed = df_processed.drop(columns=['gender'])

    # Handle Symptoms: List of symptoms usually found in this dataset
    symptoms = [
        'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness', 
        'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 
        'irritability', 'delayed_healing', 'partial_paresis', 
        'muscle_stiffness', 'alopecia', 'obesity'
    ]
    
    for col in symptoms:
        # If the raw column exists (e.g., 'polyuria'), convert it
        if col in df_processed.columns:
            new_col_name = f"{col}_Yes" # The likely name from pd.get_dummies
            # Map 'Yes'/'Positive'/'1' to 1, everything else to 0
            df_processed[new_col_name] = df_processed[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', 'positive', '1'] else 0)
            df_processed = df_processed.drop(columns=[col])

    # 3. Align with Model Features
    # This ensures columns are in the EXACT order and name the model expects.
    if hasattr(model, "feature_names_in_"):
        try:
            # We treat the model's feature names as the "source of truth"
            required_cols = list(model.feature_names_in_)
            
            # Reindex forces the dataframe to match these columns.
            # fill_value=0 ensures any missing columns are created as 0s.
            df_processed = df_processed.reindex(columns=required_cols, fill_value=0)
            
        except Exception as e:
            st.error(f"Error aligning columns: {e}")
            st.stop()
            
    return df_processed

# 1. Model Selection Dropdown
model_dir = "model"
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        st.error("No model files found. Please run train_model.py.")
        st.stop()
    selected_model_file = st.selectbox("Select a Classification Model", model_files)
else:
    st.error("Model directory not found. Please run train_model.py first.")
    st.stop()

# 2. Dataset Upload
st.write("### Upload Test Data (CSV)")
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None and selected_model_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    
    # User must select target column for evaluation
    # Default to the last column if possible
    all_cols = df.columns.tolist()
    default_idx = len(all_cols) - 1
    target_col = st.selectbox("Select Target Column (Actual Labels)", all_cols, index=default_idx)
    
    if st.button("Run Prediction"):
        # Load Model
        model_path = os.path.join(model_dir, selected_model_file)
        model = joblib.load(model_path)
        
        # Prepare Data
        # Separate Features (X) and Target (y)
        X_raw = df.drop(columns=[target_col])
        y_test = df[target_col]

        # --- APPLY FIX ---
        # Preprocess features to match training data structure
        X_test = preprocess_input(X_raw, model) 
        
        # Preprocess Target (Map text to 1/0 for metrics)
        # Handles 'Positive'/'Negative', 'Yes'/'No', 'Male'/'Female'
        mapping = {'Positive': 1, 'Negative': 0, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        if y_test.dtype == 'object':
             y_test = y_test.map(mapping).fillna(0) # Fill NaN with 0 just in case
        
        # Predict
        try:
            # Use .values to avoid feature name warnings with older sklearn versions
            X_test_array = X_test.values 
            
            y_pred = model.predict(X_test_array)
            
            # Handle probabilities safely
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test_array)[:, 1]
                except:
                    y_prob = None
            else:
                y_prob = None
            
            # 3. Display Metrics
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            # Added zero_division=0 to prevent warnings on empty classes
            col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            col1.metric("Precision", round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            
            col2.metric("Recall", round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            col2.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4))
            
            col3.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred), 4))
            if y_prob is not None and len(np.unique(y_test)) == 2:
                col3.metric("AUC Score", round(roc_auc_score(y_test, y_prob), 4))
            
            # 4. Confusion Matrix & Report
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
            
        except Exception as e:
            st.error(f"Error during prediction: {e}. Ensure feature columns match the training data.")
