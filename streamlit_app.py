import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Set Page Config
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Model Deployment")
st.write("M.Tech (AIML) - Assignment 2")

# 1. Model Selection Dropdown [cite: 92]
model_dir = "model"
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    selected_model_file = st.selectbox("Select a Classification Model", model_files)
else:
    st.error("Model directory not found. Please run train_model.py first.")
    st.stop()

# 2. Dataset Upload [cite: 91]
st.write("### Upload Test Data (CSV)")
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None and selected_model_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    
    # User must select target column for evaluation
    target_col = st.selectbox("Select Target Column (Actual Labels)", df.columns)
    
    if st.button("Run Prediction"):
        # Load Model
        model_path = os.path.join(model_dir, selected_model_file)
        model = joblib.load(model_path)
        
        # Prepare Data
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        
        # Predict
        try:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # 3. Display Metrics [cite: 93]
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            col1.metric("Precision", round(precision_score(y_test, y_pred, average='weighted'), 4))
            
            col2.metric("Recall", round(recall_score(y_test, y_pred, average='weighted'), 4))
            col2.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted'), 4))
            
            col3.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred), 4))
            if y_prob is not None and len(df[target_col].unique()) == 2:
                col3.metric("AUC Score", round(roc_auc_score(y_test, y_prob), 4))
            
            # 4. Confusion Matrix & Report [cite: 94]
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        except Exception as e:
            st.error(f"Error during prediction: {e}. Ensure feature columns match the training data.")
