import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Diabetes Predictor & Visualizer")

# Load dataset
st.subheader("Load Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data:", df.head())

    # Show data shape and stats
    st.write("Shape of dataset:", df.shape)
    st.write("Statistics:", df.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature selection
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    st.subheader("Train Model")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Feature input for prediction
    st.subheader("Predict Diabetes")
    input_data = []
    for feature in X.columns:
        value = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1)
        input_data.append(value)

    if st.button("Predict"):
        prediction = model.predict([input_data])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"The model predicts: **{result}**")

