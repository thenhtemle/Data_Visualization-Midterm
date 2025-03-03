import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import StandardScaler


""" ----------------------------- DATA CLEANING ----------------------------- """ 
# Load data
df = pd.read_csv("Data/lung_disease_data.csv")

df.info()
df.describe()

""" ----------------------------- WEB APP ----------------------------- """
# Initialize web app
st.set_page_config(
    page_title="Lung Disease Data Visualization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("Lung Disease Data Visualization")

# Sidebar


# Data

