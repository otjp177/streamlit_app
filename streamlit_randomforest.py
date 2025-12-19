import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. TITLE
# ============================================
st.title("Life Expectancy Prediction App")
st.write("Model: **Random Forest Regression**")

# ============================================
# 2. LOAD DATASET
# ============================================
DATA_PATH = "Life Expectancy Data.csv"

@st.cache_data

def load_data():
    return pd.read_csv(DATA_PATH, sep=';', engine='python', encoding='latin1', na_values=['NA', '?'])

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())
st.write(f"Jumlah baris dan kolom: {df.shape}")

# ============================================
# 3. CLEANING DATA
# ============================================
df_clean = df.dropna()
st.write("Jumlah data setelah drop NA:", df_clean.shape)

df_clean = pd.get_dummies(df_clean, drop_first=True)

# Temukan kolom target
target_col = [c for c in df_clean.columns if "life expectancy" in c.lower()][0]
st.write("Kolom Target:", target_col)

# ============================================
# 4. SPLIT DATA
# ============================================
X = df_clean.drop(target_col, axis=1)
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Jumlah data train:", X_train.shape)
st.write("Jumlah data test :", X_test.shape)

# ============================================
# 5. TRAINING RANDOM FOREST
# ============================================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
st.success("Model berhasil dilatih!")

# Prediksi
y_pred = model.predict(X_test)

# ============================================
# 6. EVALUASI MODEL
# ============================================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluasi Model")
st.write(f"MSE  : {mse}")
st.write(f"RMSE : {rmse}")
st.write(f"R²   : {r2}")

# ============================================
# 7. PLOT ACTUAL vs PREDICTED
# ============================================
st.subheader("Actual vs Predicted")

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.scatter(y_test, y_pred)
ax1.set_xlabel("Actual Life Expectancy")
ax1.set_ylabel("Predicted Life Expectancy")
ax1.set_title("Random Forest Regression: Actual vs Predicted")

# Garis y = x
x_line = np.linspace(min(y_test), max(y_test), 100)
ax1.plot(x_line, x_line)

st.pyplot(fig1)

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
st.subheader("Feature Importance")

importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(feature_names[sorted_idx], importances[sorted_idx])
ax2.set_title("Feature Importance - Random Forest Regression")
ax2.set_xlabel("Importance")

st.pyplot(fig2)
