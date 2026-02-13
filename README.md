# Machine Learning Classification Project
**M.Tech (AIML) - Assignment 2**

## 1. Problem Statement
The objective of this project is to implement and evaluate multiple machine learning classification models to predict the target variable based on a set of input features. This assignment demonstrates an end-to-end ML workflow, including data preprocessing, model training, hyperparameter tuning, and deployment using Streamlit.

The goal is to compare the performance of six different algorithms—Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), Naive Bayes, Random Forest, and XGBoost—and determine which model yields the best results for this specific dataset.

## 2. Dataset Description
* **Dataset Name:** Early Stage Diabetes Risk Prediction
* **Source:** Early Stage Diabetes Risk Prediction Dataset(2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5VG8H.
* **Description:** This dataset contains 520 instances and 16 features. It is a classification problem where the goal is to predict whether a patient is at risk of early-stage diabetes (Positive/Negative) based on their symptoms and demographic profile.
 
## 3. Key Features
The dataset includes the following features:
* **Demographic:** Age (Integer), Gender (Categorical - Male/Female)
* **Symptoms (Binary - Yes/No):**
    * **Polyuria:** Excessive urination
    * **Polydipsia:** Excessive thirst
    * **Sudden Weight Loss**
    * **Weakness**
    * **Polyphagia:** Excessive hunger
    * **Genital Thrush**
    * **Visual Blurring**
    * **Itching**
    * (And additional clinical features typically included in the full dataset)

## 3. Model Comparison Table
The following table summarizes the evaluation metrics for all implemented models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9231 | 0.9774 | 0.9225 | 0.9231 | 0.9224 | 0.8204 |
| **Decision Tree** | 0.9519 | 0.9648 | 0.9582 | 0.9519 | 0.9527 | 0.8985 |
| **KNN** | 0.8942 | 0.9774 | 0.9022 | 0.8942 | 0.8960 | 0.7698 |
| **Naive Bayes** | 0.9135 | 0.9607 | 0.9129 | 0.9135 | 0.9131 | 0.7988 |
| **Random Forest (Ensemble)** | 0.9904 | 1.0000 | 0.9907 | 0.9904 | 0.9904 | 0.9782 |
| **XGBoost (Ensemble)** | 0.9712 | 1.0000 | 0.9736 | 0.9712 | 0.9715 | 0.9370 |

## 4. Observations on Performance
Based on the evaluation metrics above, here are the observations for each model:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed well as a baseline with **92.3% accuracy**. The high AUC of **0.977** indicates it is very effective at ranking positive instances, even if the decision boundary isn't perfect. |
| **Decision Tree** | Achieved strong performance (**95.2% accuracy**) with a high MCC of **0.899**, showing it captured non-linear patterns better than the linear baseline. |
| **KNN** | This was the lowest performing model (**89.4% accuracy**), likely due to the high dimensionality or scaling sensitivity, though it still maintained a respectable AUC. |
| **Naive Bayes** | Performed reliably (**91.3% accuracy**) given its simplicity. It showed slightly lower precision/recall balance compared to tree-based models. |
| **Random Forest (Ensemble)** | **Best Performing Model.** It achieved **99.04% accuracy** and a **perfect AUC of 1.0**. The near-perfect F1 score (0.99) proves it handled both false positives and false negatives exceptionally well. |
| **XGBoost (Ensemble)** | Excellent performance (**97.1% accuracy**) and a **perfect AUC of 1.0**. While slightly lower in accuracy than Random Forest, it remains a highly robust model for this dataset. |

## 5. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run streamlit_app.py`
