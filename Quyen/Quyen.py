import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as skp  # Added for encoding in Data Analysis

### ------------------------ PAGE INITIALIZATION ------------------------
st.set_page_config(page_title="Lung Cancer", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

PALLETE = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

st.title("Lung Cancer Data Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Introduction", "The Raw Data", "Summary Statistics", "Data Analysis"]
selected_section = st.sidebar.radio("Go to:", sections)

# Load the dataset (used across sections)
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("lung_disease_data_cleaned.csv")
data = load_data()

# Function to encode categorical data for analysis
def encode_data(data):
    encoder = skp.LabelEncoder()
    encoded_data = data.copy()
    for col in data.select_dtypes(include=['object']).columns:
        encoded_data[col] = encoder.fit_transform(data[col])
    return encoded_data

# Display content based on selected section
if selected_section == "Introduction":
    st.header("Introduction")
    st.markdown("""
    This dataset captures detailed information about patients suffering from various lung conditions. It includes: \n 
        🧑‍🤝‍🧑 Age & Gender: Patient demographics to understand the spread across age groups and gender.
        🚬 Smoking Status: Whether the patient is a smoker or non-smoker.
        🌡️ Lung Capacity: Measured lung function to assess disease severity.
        🫁 Disease Type: The specific lung condition, like COPD or Bronchitis.
        💊 Treatment Type: Different treatments patients received, including therapy, medication, or surgery.
        🏥 Hospital Visits: Number of visits to the hospital for managing the condition.
        ✅ Recovery Status: Indicates whether the patient recovered after treatment.""")

elif selected_section == "The Raw Data":
    st.header("The Raw Data")
    st.dataframe(data)

elif selected_section == "Summary Statistics":
    st.header("Summary Statistics")
    st.write("Summary Statistics of the Dataset:")
    st.dataframe(data.describe())

    st.header("Distribution of Numerical Features:")
    num_col = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_select = st.selectbox("Select a numerical feature:", num_col)

    if num_select:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[num_select], kde=True, color=PALLETE[0], ax=ax)
        ax.set_title(f"Distribution of {num_select}")
        st.pyplot(fig)

    st.header("Distribution of Categorical Features:")
    cat_col = data.select_dtypes(include=['object']).columns.tolist()
    cat_select = st.selectbox("Select a categorical feature:", cat_col)

    if cat_select:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data[cat_select], palette=PALLETE, ax=ax)
        ax.set_title(f"Distribution of {cat_select}")
        st.pyplot(fig)

elif selected_section == "Data Analysis":
    st.header("Data Analysis")
    encoded_data = encode_data(data)  # Encode categorical data for analysis

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("This heatmap shows the correlation between numerical features.")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = encoded_data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Between Numerical Features")
    st.pyplot(fig)

    st.write(f"""🔍 **Insight:** Weak Inter-Feature Correlations: \n
    - Most correlations are near zero (e.g., 0.00 to -0.03), indicating little to no linear relationship between features.
    - Age, Lung Capacity, and Hospital Visits remain weakly correlated with each other and with categorical variables.
    - Smoking Status, Disease Type, Treatment Type, and Gender show negligible correlations with numerical features and other categoricals.
    - Recovered has a very weak negative correlation with most features (e.g., -0.02 with Smoking Status), suggesting no strong predictive relationship.""")

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    num_col = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    feature_x = st.selectbox("Select a numerical feature for X-axis:", num_col)
    feature_y = st.selectbox("Select a numerical feature for Y-axis:", num_col)
    plot_type = st.radio("Plot type:", ["Scatter", "Hexbin", "2D KDE"])

    if feature_x != feature_y:
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == "Scatter":
            sns.scatterplot(x=data[feature_x], y=data[feature_y], hue=data["Recovered"], palette=PALLETE[:2], ax=ax)
        elif plot_type == "Hexbin":
            hb = ax.hexbin(data[feature_x], data[feature_y], gridsize=30, cmap="Blues", mincnt=1)
            fig.colorbar(hb, ax=ax, label="Count")
        elif plot_type == "2D KDE":
            sns.kdeplot(x=data[feature_x], y=data[feature_y], cmap="Blues", fill=True, ax=ax)
        
        ax.set_title(f"{feature_x} vs {feature_y} ({plot_type})")
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        plt.tight_layout()
        st.pyplot(fig)
        st.write(f"🔍 **Insight:** {plot_type} plot shows the density of data points. Hexbin and KDE help with overplotting.")
    else:
        st.warning("Please select different features for X and Y axes.")

    # Recovery Rate by Factors
    st.subheader("Recovery Rate Analysis")
    factor = st.selectbox("Select a factor to compare recovery rates:", ['Smoking Status', 'Disease Type', 'Treatment Type'])

    def rec_rate(data, factor):
        # Calculate recovery rate as proportions of "Yes" and "No" for each category
        recovery_rate = data.groupby(factor)["Recovered"].value_counts(normalize=True).unstack().fillna(0)
        return recovery_rate

    # Get recovery rates
    recovery_data = rec_rate(data, factor)

    # Plot the recovery rates as a stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    recovery_data.plot(kind='bar', stacked=True, color=PALLETE[:2], ax=ax)
    ax.set_title(f"Recovery Rate by {factor}")
    ax.set_xlabel(factor)
    ax.set_ylabel("Proportion")
    ax.legend(title="Recovered", labels=["No", "Yes"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Display the recovery rate table and insight
    st.write("### Recovery Rate Breakdown:")
    st.dataframe(recovery_data.style.format("{:.2%}"))  # Display as percentages
    st.write(f"🔍 **Insight:** The chart and table show the proportion of patients who recovered (Yes) vs. did not recover (No) based on {factor}. "
             "This helps identify which categories are associated with higher or lower recovery rates.")