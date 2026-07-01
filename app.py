import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="E-Commerce Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 E-Commerce Customer Churn Prediction Dashboard")

st.markdown("""
Predict whether a customer is likely to churn using the trained Random Forest model.
""")

model = joblib.load("outputs/best_model.joblib")

st.header("Dataset Overview")

data = pd.read_excel(
    "data/E Commerce Dataset.xlsx",
    sheet_name="E Comm"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(data))

with col2:
    st.metric(
        "Churn Rate",
        f"{data['Churn'].mean()*100:.2f}%"
    )

with col3:
    st.metric(
        "Features",
        len(data.columns)-1
    )

uploaded_file = st.file_uploader(
    "Upload customer data",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if st.button("Predict Churn"):
        # Create a copy for prediction
        prediction_df = df.copy()

        # Remove columns not used during training
        for col in ["CustomerID", "Churn"]:
            if col in prediction_df.columns:
                prediction_df = prediction_df.drop(columns=[col])

        predictions = model.predict(prediction_df)
        probabilities = model.predict_proba(prediction_df)[:, 1]

        df["Churn Prediction"] = predictions
        df["Churn Probability"] = (
            probabilities * 100
        ).round(2)

        st.subheader("Prediction Results")
        st.dataframe(df)
        st.subheader("Prediction Summary")

        high_risk = (df["Churn Prediction"] == 1).sum()
        low_risk = (df["Churn Prediction"] == 0).sum()

        c1, c2 = st.columns(2)

        with c1:
            st.metric("High Risk Customers", high_risk)

        with c2:
            st.metric("Low Risk Customers", low_risk)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "outputs/confusion_matrix.png",
        caption="Confusion Matrix"
    )

with col2:
    st.image(
        "outputs/feature_importance.png",
        caption="Feature Importance"
    )

col3, col4 = st.columns(2)

with col3:
    st.image(
        "outputs/roc_curve.png",
        caption="ROC Curve"
    )

with col4:
    st.image(
        "outputs/pr_curve.png",
        caption="Precision Recall Curve"
    )

