import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Title
# ===============================
st.title("Life Expectancy Regression App")
st.write("Model: **Lasso Regression**")

# ===============================
# Load Dataset
# ===============================
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

# ===============================
# Missing Value
# ===============================
st.subheader("Missing Value")
st.write(df.isnull().sum())

# ===============================
# Cleaning Data
# ===============================
df_clean = df.dropna()
st.write(f"Jumlah data setelah drop NA: {df_clean.shape}")

# One-hot encoding
df_clean = pd.get_dummies(df_clean, drop_first=True)

# ===============================
# Target Column
# ===============================
target_col = [c for c in df_clean.columns if "life expectancy" in c.lower()][0]
st.write(f"Target Kolom: **{target_col}**")

# ===============================
# Split Data
# ===============================
X = df_clean.drop(target_col, axis=1)
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("### Jumlah Data Train:", X_train.shape)
st.write("### Jumlah Data Test:", X_test.shape)

# ===============================
# Model Training
# ===============================
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)
st.success("Model berhasil dilatih!")

# ===============================
# Prediction & Evaluation
# ===============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluation Metrics")
st.write(f"**MAE:** {mae}")
st.write(f"**MSE:** {mse}")
st.write(f"**RMSE:** {rmse}")
st.write(f"**R² Score:** {r2}")

# ===============================
# Plot Actual vs Predicted
# ===============================
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Life Expectancy")
ax.set_ylabel("Predicted Life Expectancy")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# ===============================
# Feature Importance
# ===============================
st.subheader("Feature Importance")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.write(importance.head(10))

fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.barplot(data=importance.head(20), x='Importance', y='Feature', ax=ax2)
st.pyplot(fig2)
