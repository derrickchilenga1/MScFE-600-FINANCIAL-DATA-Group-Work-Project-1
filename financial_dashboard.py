# !pip install streamlit pandas matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
st.title("Financial Data Dashboard")
st.write("Interactive dashboard for exploring and modeling financial data.")

# Replace with your actual file or data source
data_file = "your_financial_data.csv"
data = pd.read_csv(data_file)  # Make sure your data file is present

# --- Sidebar Navigation ---
section = st.sidebar.radio(
    "Choose Section",
    ["Project Details", "EDA", "Model Training & Prediction", "Predict on New Data"]
)

# --- Project Details ---
if section == "Project Details":
    st.header("Project Overview")
    st.markdown("""
        **Objective:**  
        Analyze financial time series data for yield curve modeling, ETF return analysis, and more.

        **Tasks:**  
        - Data quality checks  
        - Yield curve modeling (Nelson-Siegel, Cubic-Spline)  
        - PCA on yield data  
        - ETF return analysis  
        - Visualization and interpretation
    """)

# --- EDA Section ---
elif section == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("#### Data Preview")
    st.dataframe(data.head())

    st.write("#### Data Summary")
    st.write(data.describe())

    st.write("#### Null Values")
    st.write(data.isnull().sum())

    # Visualize a selected feature
    feature = st.sidebar.selectbox("Feature to visualize", data.columns)
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x=data.index, y=feature, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.write("#### Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# --- Model Training & Prediction ---
elif section == "Model Training & Prediction":
    st.header("Model Training & Prediction")
    st.write("Train a classifier or regression model on your dataset.")

    # Select target and features
    target = st.sidebar.selectbox("Target Variable", data.columns)
    features = st.sidebar.multiselect(
        "Features (inputs)", [col for col in data.columns if col != target]
    )

    if features:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        X = data[features].dropna()
        y = data[target].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("#### Model Performance")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

        # Feature importance
        st.write("#### Feature Importances")
        imp = pd.Series(model.feature_importances_, index=features)
        st.bar_chart(imp)

# --- Predict on New Data ---
elif section == "Predict on New Data":
    st.header("Predict on New Data")
    st.write("Input features below to get a prediction from the model.")

    # Example for three features; adjust based on your data
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)
    feature3 = st.number_input("Feature 3", value=0.0)

    # Load trained model (in practice, save/load with joblib or pickle)
    # Here, retrain for demo purposes
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    # Dummy train to avoid errors; replace with your actual training
    if "features" in locals() and features:
        X = data[features].dropna()
        y = data[target].loc[X.index]
        model.fit(X, y)

        if st.button("Predict"):
            input_data = np.array([[feature1, feature2, feature3]])
            pred = model.predict(input_data)[0]
            st.write(f"**Predicted Value:** {pred:.2f}")

