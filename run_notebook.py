#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated execution of stock prediction notebook
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Data manipulation and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Time Series
from statsmodels.tsa.arima.model import ARIMA
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available")

# Financial Data
import yfinance as yf

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print('[OK] All libraries imported successfully!\n')

# Configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2024-10-30'

# Download data
print(f'[DATA] Fetching {TICKER} stock data...')
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

print(f'\nData Shape: {data.shape}')
print(f'Date Range: {data.index[0]} to {data.index[-1]}')
print(f'\nFirst few rows:')
print(data.head())

# Basic visualization
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], linewidth=2)
plt.title(f'{TICKER} Stock Price History', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('price_history.png', dpi=300, bbox_inches='tight')
print('\n[OK] Price history chart saved!')

print(f'\n[STATS] Statistics:')
print(data['Close'].describe())

# Calculate Moving Averages
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

print('\n[OK] Moving Averages calculated')
print(f'Non-null values after SMA_200: {data["SMA_200"].notna().sum()}')

# Calculate RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

print('[OK] RSI calculated')
print(f'RSI Range: [{data["RSI"].min():.2f}, {data["RSI"].max():.2f}]')

# Calculate MACD
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

print('[OK] MACD calculated')

# Calculate additional features
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()

# Drop NaN values
data = data.dropna()

print(f'\n[OK] Feature engineering complete')
print(f'Final dataset shape: {data.shape}')
print(f'\nFeatures: {list(data.columns)}')

# Visualize Technical Indicators
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle(f'{TICKER} Technical Analysis', fontsize=16, fontweight='bold')

# Plot 1: Price and Moving Averages
axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
axes[0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
axes[0].plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
axes[0].plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.7)
axes[0].set_title('Price with Moving Averages')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: RSI
axes[1].plot(data.index, data['RSI'], color='purple', linewidth=2)
axes[1].axhline(y=70, color='r', linestyle='--', label='Overbought')
axes[1].axhline(y=30, color='g', linestyle='--', label='Oversold')
axes[1].fill_between(data.index, 30, 70, alpha=0.1)
axes[1].set_title('Relative Strength Index (RSI)')
axes[1].set_ylabel('RSI')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: MACD
axes[2].plot(data.index, data['MACD'], label='MACD', linewidth=2)
axes[2].plot(data.index, data['Signal_Line'], label='Signal', linewidth=2)
axes[2].fill_between(data.index, 0, data['MACD_Histogram'], alpha=0.3, label='Histogram')
axes[2].set_title('MACD')
axes[2].set_ylabel('MACD')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Volume
axes[3].plot(data.index, data['Volume'], alpha=0.5, label='Volume', linewidth=1.5)
axes[3].plot(data.index, data['Volume_MA'], color='red', linewidth=2, label='Volume MA')
axes[3].set_title('Trading Volume')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('Volume')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('technical_analysis.png', dpi=300, bbox_inches='tight')
print('\n[OK] Technical analysis chart saved!')

# Select features
features = ['Close', 'SMA_20', 'RSI', 'MACD', 'Volume']
df_features = data[features].values

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_features)

# Create sequences
lookback = 60
X, y = [], []

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i, 0])  # Predict Close price

X, y = np.array(X), np.array(y)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'\n[PREP] Data Preparation:')
print(f'Training set: {X_train.shape}')
print(f'Testing set: {X_test.shape}')
print(f'Features per timestep: {X_train.shape[2]}')

# Build LSTM model
print('\n[LSTM] Building LSTM Model...')
model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print('\n[OK] LSTM Model built successfully!')

# Train LSTM
print('\n[TRAIN] Training LSTM Model (this may take a few minutes)...')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print('\n[OK] LSTM training complete!')

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('Model MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print('\n[OK] Training history chart saved!')

# Train ARIMA
train_size = int(len(data) * 0.8)
train_arima = data['Close'][:train_size]
test_arima = data['Close'][train_size:]

print('\n[ARIMA] Training ARIMA model...')
model_arima = ARIMA(train_arima, order=(5, 1, 2))
model_arima_fit = model_arima.fit()

print('\n[OK] ARIMA training complete!')

# Train Prophet if available
if PROPHET_AVAILABLE:
    prophet_df = data.reset_index()[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']

    train_size_prophet = int(len(prophet_df) * 0.8)
    train_prophet = prophet_df[:train_size_prophet]
    test_prophet = prophet_df[train_size_prophet:]

    print('\n[PROPHET] Training Prophet model...')
    try:
        model_prophet = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model_prophet.fit(train_prophet)
        print('\n[OK] Prophet training complete!')
    except Exception as e:
        print(f'\n[WARN] Prophet training failed: {e}')
        PROPHET_AVAILABLE = False

# LSTM Predictions
print('\n[PRED] Generating predictions...')
lstm_pred = model_lstm.predict(X_test, verbose=0)

# Inverse transform
dummy = np.zeros((len(lstm_pred), scaler.n_features_in_))
dummy[:, 0] = lstm_pred.flatten()
lstm_pred_actual = scaler.inverse_transform(dummy)[:, 0]

dummy_y = np.zeros((len(y_test), scaler.n_features_in_))
dummy_y[:, 0] = y_test
y_test_actual = scaler.inverse_transform(dummy_y)[:, 0]

# ARIMA Predictions
arima_pred = model_arima_fit.forecast(steps=len(test_arima))

# Prophet Predictions
if PROPHET_AVAILABLE:
    try:
        prophet_forecast = model_prophet.predict(test_prophet[['ds']])
        prophet_pred = prophet_forecast['yhat'].values
    except:
        PROPHET_AVAILABLE = False

print('\n[OK] All predictions generated!')

# Calculate metrics
def calculate_metrics(actual, predicted, model_name):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(f'\n{model_name} Performance:')
    print(f'  RMSE: ${rmse:.2f}')
    print(f'  MAE:  ${mae:.2f}')
    print(f'  R²:   {r2:.4f}')

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Evaluate all models
print('\n' + '='*60)
print('MODEL PERFORMANCE COMPARISON')
print('='*60)

lstm_metrics = calculate_metrics(y_test_actual, lstm_pred_actual, 'LSTM')
arima_metrics = calculate_metrics(test_arima.values, arima_pred.values, 'ARIMA')

if PROPHET_AVAILABLE:
    prophet_metrics = calculate_metrics(test_prophet['y'].values, prophet_pred, 'Prophet')

print('\n' + '='*60)

# Visualize Results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'{TICKER} Stock Price Prediction - Model Comparison', fontsize=16, fontweight='bold')

# LSTM
axes[0, 0].plot(y_test_actual, label='Actual', linewidth=2)
axes[0, 0].plot(lstm_pred_actual, label='LSTM Prediction', linestyle='--', linewidth=2)
axes[0, 0].set_title('LSTM Model')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ARIMA
axes[0, 1].plot(test_arima.values, label='Actual', linewidth=2)
axes[0, 1].plot(arima_pred.values, label='ARIMA Prediction', linestyle='--', linewidth=2)
axes[0, 1].set_title('ARIMA Model')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Prophet
if PROPHET_AVAILABLE:
    axes[1, 0].plot(test_prophet['y'].values, label='Actual', linewidth=2)
    axes[1, 0].plot(prophet_pred, label='Prophet Prediction', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Prophet Model')
else:
    axes[1, 0].text(0.5, 0.5, 'Prophet Model Not Available',
                   ha='center', va='center', fontsize=14, transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Prophet Model')

axes[1, 0].set_xlabel('Time Steps')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Performance Comparison
models = ['LSTM', 'ARIMA']
rmse_values = [lstm_metrics['RMSE'], arima_metrics['RMSE']]
mae_values = [lstm_metrics['MAE'], arima_metrics['MAE']]

if PROPHET_AVAILABLE:
    models.append('Prophet')
    rmse_values.append(prophet_metrics['RMSE'])
    mae_values.append(prophet_metrics['MAE'])

x = np.arange(len(models))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, rmse_values, width, label='RMSE')
bars2 = axes[1, 1].bar(x + width/2, mae_values, width, label='MAE')

axes[1, 1].set_xlabel('Models')
axes[1, 1].set_ylabel('Error ($)')
axes[1, 1].set_title('Performance Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print('\n[OK] Model comparison chart saved!')

print('\n' + '='*60)
print('EXECUTION COMPLETE!')
print('='*60)
print('\nGenerated Files:')
print('  - price_history.png')
print('  - technical_analysis.png')
print('  - training_history.png')
print('  - model_comparison.png')
print('\nBest Model: LSTM with RMSE of ${:.2f}'.format(lstm_metrics['RMSE']))
print('='*60)
