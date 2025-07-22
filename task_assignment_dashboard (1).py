
import streamlit as st
import pandas as pd
import joblib

# Load models and vectorizer
rf_model = joblib.load("random_forest_classifier.pkl")
xgb_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

# App title
st.title("🚀 AI-Powered Task Assignment Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your task dataset (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset")
    st.dataframe(df)

    required_columns = {'task_description', 'assigned_user', 'deadline', 'current_user_load'}
    if required_columns.issubset(df.columns):
        # Priority Prediction
        st.subheader("⚙️ Predict Task Priorities")

        X_priority = priority_vectorizer.transform(df['task_description'].astype(str))
        predicted_priorities = xgb_model.predict(X_priority)
        decoded_priorities = priority_label_encoder.inverse_transform(predicted_priorities)
        df['Predicted Priority'] = decoded_priorities

        # Task Type Prediction
        st.subheader("📌 Predict Task Types")
        X_features = df[['deadline', 'current_user_load']].copy()
        X_features['deadline'] = pd.to_datetime(X_features['deadline'], errors='coerce')
        X_features['deadline_days'] = (X_features['deadline'] - pd.Timestamp.today()).dt.days
        X_features = X_features[['deadline_days', 'current_user_load']].fillna(0)

        task_types = rf_model.predict(X_features)
        df['Predicted Task Type'] = task_types

        st.success("✅ Predictions Complete")
        st.dataframe(df[['task_description', 'Predicted Priority', 'Predicted Task Type']])
    else:
        st.error("❌ Required features not found in the uploaded dataset. Please include: task_description, assigned_user, deadline, current_user_load")
