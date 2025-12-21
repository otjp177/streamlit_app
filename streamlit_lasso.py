import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

st.title("Life Expectancy Prediction App")
st.write("Model: **Lasso Regression**")

DATA_PATH = "Life Expectancy Data.csv"
@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        sep=';',
        engine='python',
        encoding='latin1',
        na_values=['NA', '?']
    )

df = load_data()
st.subheader("Dataset Original")
st.write(df.head())
st.write(f"Jumlah baris dan kolom: {df.shape}")

boston = fetch_openml(name="boston", version=1, as_frame=True)

X = housing_ames.data
y = housing_ames.target

load_data