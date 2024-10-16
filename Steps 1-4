import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sidebar for file uploads
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded dataset
    data = pd.read_csv(uploaded_file)

    # Remove duplicates
    data_cleaned = data.drop_duplicates()

    # Main section for displaying and processing the selected dataset
    st.title("Medical Insurance Dataset Analysis")

    # Step 1: Display a sample of the dataset
    st.write("### Sample Data from the Dataset")
    st.dataframe(data_cleaned.sample(5), use_container_width=True)

    # Check if "Unnamed: 0" column exists and drop it
    if "Unnamed: 0" in data_cleaned.columns:
        data_cleaned = data_cleaned.drop(columns=["Unnamed: 0"])
        st.write("Dropped 'Unnamed: 0' column from the dataset.")
        st.dataframe(data_cleaned.sample(5), use_container_width=True)

    # Display data types
    st.write("### Cleaned Data Types")
    dtype_df = pd.DataFrame(data_cleaned.dtypes, columns=["Data Type"]).reset_index().rename(
        columns={"index": "Column Name"})
    st.dataframe(dtype_df, use_container_width=True)

    # Display the shape and columns of the dataset
    st.write("### Dataset Information")
    st.write(f"### Shape: `{data_cleaned.shape}`")
    st.write("Columns in the dataset:", data_cleaned.columns.tolist())

    # Step 2: Problem Statement Definition
    target = 'charges'  # For this dataset, we assume 'charges' is the target variable
    st.write(f"### Target Variable: `{target}`")

    # Step 3 - Visualizing the Target Variable
    st.write("## Step 3: Visualizing the Target Variable")

    fig, ax = plt.subplots()
    ax.hist(data_cleaned[target], bins=30, edgecolor='k', alpha=0.7)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Step 4: Basic Data Exploration
    st.write("## Step 4: Data Exploration")

    # Create two columns for data types and summary statistics
    col1, col2 = st.columns(2)

    # Column 1: Data Types
    with col1:
        st.write("### Data Types:")
        st.dataframe(dtype_df, use_container_width=True)

    # Column 2: Summary Statistics
    with col2:
        st.write("### Summary Statistics:")
        st.dataframe(data_cleaned.describe(), use_container_width=True)

    # Pairplot for numeric columns
    numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        st.write("## Pairplot of Numeric Variables")
        sns.pairplot(data_cleaned[numeric_cols])
        st.pyplot(plt.gcf())
    else:
        st.write("Not enough numeric variables for a pairplot.")
else:
    st.write("Please upload a CSV file to proceed.")
