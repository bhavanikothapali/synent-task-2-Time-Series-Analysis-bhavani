import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# TITLE

st.title("Bitcoin Time Series Analysis")

st.write("Trend Analysis and Forecasting using Random Forest")

# LOAD DATA

df = pd.read_csv("bitcoin.csv")

# SHOW DATA

st.subheader("Dataset Preview")

st.dataframe(df.head())

# PREPROCESSING

df['Bitcoin_Price'] = df['Bitcoin_Price'].replace(
    r'[\\$,]',
    '',
    regex=True
).astype(float)

# FEATURE ENGINEERING

df['BTC_Lag1'] = df['Bitcoin_Price'].shift(1)

df['BTC_7Day_Avg'] = df['Bitcoin_Price'].rolling(window=7).mean()

df.dropna(inplace=True)

# FEATURES

x = df[
    [
        'Bitcoin_Vol.',
        'Google_Price',
        'S&P_500_Price',
        'Nasdaq_100_Price'
    ]
]

y = df['Bitcoin_Price']

# TRAIN TEST SPLIT

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.4,
    random_state=42
)

# LINEAR REGRESSION

model = LinearRegression()

model.fit(x_train, y_train)

# RANDOM FOREST

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(x_train, y_train)

# PREDICTIONS

rf_preds = rf_model.predict(x_test)

# METRICS

st.subheader("Model Performance")

st.write(
    f"Random Forest R2 Score: {r2_score(y_test, rf_preds):.2f}"
)

st.write(
    f"Random Forest MAE: {mean_absolute_error(y_test, rf_preds):.2f}"
)

# ACTUAL VS PREDICTED

st.subheader("Actual vs Predicted Bitcoin Prices")

plt.figure(figsize=(10,5))

plt.plot(
    y_test.values[:50],
    label='Actual',
    color='blue'
)

plt.plot(
    rf_preds[:50],
    label='Predicted (RF)',
    color='red',
    linestyle='--'
)

plt.title('Actual vs Predicted Bitcoin Prices')

plt.legend()

plt.grid(True)

st.pyplot(plt)

# FEATURE IMPORTANCE

st.subheader("Feature Importance Analysis")

importances = rf_model.feature_importances_

feature_names = x.columns

plt.figure(figsize=(10,6))

plt.barh(
    feature_names,
    importances,
    color='skyblue'
)

plt.xlabel('Importance Score')

plt.title(
    'Most Important Features for Bitcoin Price'
)

st.pyplot(plt)

# INSIGHTS

st.subheader("Insights")

st.write("""
1. Bitcoin price trends were analyzed using machine learning models.

2. Random Forest performed well for prediction.

3. Feature importance analysis identified key influencing factors.

4. Historical market indicators impact Bitcoin price movement.
""")
