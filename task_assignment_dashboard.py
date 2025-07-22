
import streamlit as st
import pandas as pd
import joblib

# Load saved models
priority_model = joblib.load('priority_xgboost.pkl')
priority_encoder = joblib.load('priority_label_encoder.pkl')

st.set_page_config(page_title="AI Task Assignment", layout="wide")
st.title("🧠 AI-Powered Task Assignment Dashboard")

# Upload CSV file
task_file = st.file_uploader("📂 Upload Your Task Dataset (CSV Format)", type=['csv'])

if task_file:
    df = pd.read_csv(task_file)

    st.subheader("📋 Uploaded Task Preview")
    st.dataframe(df.head())

    required_features = ['task_length', 'has_keyword_urgent', 'is_weekend_deadline',
                         'deadline_days_remaining', 'user_current_load', 'past_behavior_score']

    if all(feature in df.columns for feature in required_features):

        # Predict priority
        X_priority = df[required_features]
        y_pred = priority_model.predict(X_priority)
        df['predicted_priority'] = priority_encoder.inverse_transform(y_pred)

        st.success("✅ Priority Predicted Successfully!")

        st.subheader("📊 Predicted Priorities Overview")
        st.dataframe(df[['task_id', 'task_description', 'predicted_priority']])

        # Simulate users data
        users_df = df[['assigned_user', 'user_current_load', 'past_behavior_score']].drop_duplicates()

        # Task assignment logic
        def assign_tasks(df, users_df):
            df['final_assigned_user'] = None
            for idx, row in df.iterrows():
                candidates = users_df.sort_values(
                    by=['user_current_load', 'past_behavior_score'],
                    ascending=[True, False]
                )
                selected_user = candidates.iloc[0]['assigned_user']
                df.at[idx, 'final_assigned_user'] = selected_user
                users_df.loc[users_df['assigned_user'] == selected_user, 'user_current_load'] += row['workload']
            return df

        assigned_df = assign_tasks(df.copy(), users_df.copy())

        st.subheader("📌 Final Assigned Tasks")
        st.dataframe(assigned_df[['task_id', 'task_description', 'predicted_priority', 'final_assigned_user']])

        # Download button
        csv_output = assigned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Assigned Tasks CSV", data=csv_output,
                           file_name="assigned_tasks.csv", mime='text/csv')

    else:
        st.error("❌ Required features not found in the uploaded dataset. Please include:
" +
                 ", ".join(required_features))
else:
    st.info("📎 Please upload your task CSV file to begin.")
