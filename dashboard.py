import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title and description
st.title("Financial Data Dashboard")
st.write("Visualize and analyze financial time series data.")

# Load your processed data (replace with your actual CSV or data path)
# Example: df = pd.read_csv('your_processed_data.csv', parse_dates=['DATE'], index_col='DATE')
# For demonstration, let's create a dummy DataFrame
df = pd.DataFrame({
    "DATE": pd.date_range(start="2014-09-04", periods=5, freq="D"),
    "CreditCardDelinq": [None, None, None, None, None],
    "AvgMortgageRate": [4.1, None, None, None, None],
    "BuildingMaterialPx": [None, None, None, None, None],
    "SP500": [None, None, None, None, None],
    "InvestmentBondYield": [4.03, 4.10, 4.08, 4.09, 4.13],
    "TreasuryBondYield": [3.205, 3.237, 3.223, 3.233, 3.269],
})
df.set_index("DATE", inplace=True)

# Sidebar for feature selection
feature = st.sidebar.selectbox(
    "Choose a feature to visualize",
    df.columns
)

# Line chart of the selected feature
st.line_chart(df[feature])

# Show data
if st.checkbox("Show raw data"):
    st.write(df)

# You can add more plots, metrics, and widgets as needed!
