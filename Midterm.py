import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import StandardScaler

# Initialize web app
st.set_page_config(
    page_title="Lung Disease Data Visualization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

### ----------------------------- DATA CLEANING ----------------------------- 
# Load data
df = pd.read_csv("Data/lung_disease_data.csv")

df.info()
df.describe()

### ----------------------------- WEB APP ----------------------------- 
# Title
st.title("Lung Disease Data Visualization")

# Raw data 
st.subheader("Raw Data")
st.write(df)

# Data Visualization

#### Distribution of Age
st.subheader("Distribution of Age")
age_distribution = df["Age"].value_counts()
st.write(age_distribution)


