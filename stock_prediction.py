"""
Stock Market Trend Prediction Using Machine Learning
Combines LSTM, ARIMA, and Prophet models with technical indicators
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Financial Data
import yfinance as yf

class StockPredictor:
    """
    A comprehensive stock prediction system combining multiple forecasting methods
    """
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.arima_model = None
        self.prophet_model = None
        
    def fetch_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def calculate_technical_indicators(self):
        """Calculate SMA, RSI, and MACD technical indicators"""
        df = self.data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Additional features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        self.data = df
        print(f"Technical indicators calculated. Shape: {df.shape}")
        return df
    
    def prepare_lstm_data(self, lookback=60, features=['Close', 'SMA_20', 'RSI', 'MACD', 'Volume']):
        """Prepare data for LSTM model"""
        df = self.data[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape, units=[128, 64, 32], dropout=0.2):
        """Build and compile LSTM model with hyperparameter tuning"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        model.add(LSTM(units=units[1], return_sequences=True))
        model.add(Dropout(dropout))
        
        # Third LSTM layer
        model.add(LSTM(units=units[2], return_sequences=False))
        model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile with optimized learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train LSTM model with early stopping"""
        print("\n=== Training LSTM Model ===")
        
        self.lstm_model = self.build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=[128, 64, 32],
            dropout=0.2
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def train_arima(self, order=(5, 1, 2)):
        """Train ARIMA model"""
        print("\n=== Training ARIMA Model ===")
        
        train_size = int(len(self.data) * 0.8)
        train_data = self.data['Close'][:train_size]
        
        self.arima_model = ARIMA(train_data, order=order)
        self.arima_model = self.arima_model.fit()
        
        print(f"ARIMA{order} model trained")
        return self.arima_model
    
    def train_prophet(self):
        """Train Prophet model"""
        print("\n=== Training Prophet Model ===")
        
        # Prepare data for Prophet
        prophet_df = self.data.reset_index()[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        train_size = int(len(prophet_df) * 0.8)
        train_prophet = prophet_df[:train_size]
        
        self.prophet_model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        self.prophet_model.fit(train_prophet)
        print("Prophet model trained")
        return self.prophet_model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and compare performance"""
        results = {}
        
        # LSTM Predictions
        lstm_predictions = self.lstm_model.predict(X_test)
        
        # Inverse transform to get actual prices
        dummy = np.zeros((len(lstm_predictions), self.scaler.n_features_in_))
        dummy[:, 0] = lstm_predictions.flatten()
        lstm_predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        dummy_actual = np.zeros((len(y_test), self.scaler.n_features_in_))
        dummy_actual[:, 0] = y_test
        y_test_actual = self.scaler.inverse_transform(dummy_actual)[:, 0]
        
        results['LSTM'] = {
            'predictions': lstm_predictions,
            'actual': y_test_actual,
            'rmse': np.sqrt(mean_squared_error(y_test_actual, lstm_predictions)),
            'mae': mean_absolute_error(y_test_actual, lstm_predictions),
            'r2': r2_score(y_test_actual, lstm_predictions)
        }
        
        # ARIMA Predictions
        train_size = int(len(self.data) * 0.8)
        test_data = self.data['Close'][train_size:]
        arima_predictions = self.arima_model.forecast(steps=len(test_data))
        
        results['ARIMA'] = {
            'predictions': arima_predictions.values,
            'actual': test_data.values,
            'rmse': np.sqrt(mean_squared_error(test_data.values, arima_predictions.values)),
            'mae': mean_absolute_error(test_data.values, arima_predictions.values),
            'r2': r2_score(test_data.values, arima_predictions.values)
        }
        
        # Prophet Predictions (if model was trained)
        if self.prophet_model is not None:
            try:
                prophet_df = self.data.reset_index()[['Date', 'Close']].copy()
                prophet_df.columns = ['ds', 'y']
                train_size_prophet = int(len(prophet_df) * 0.8)
                test_prophet = prophet_df[train_size_prophet:]

                prophet_forecast = self.prophet_model.predict(test_prophet[['ds']])
                prophet_predictions = prophet_forecast['yhat'].values

                results['Prophet'] = {
                    'predictions': prophet_predictions,
                    'actual': test_prophet['y'].values,
                    'rmse': np.sqrt(mean_squared_error(test_prophet['y'].values, prophet_predictions)),
                    'mae': mean_absolute_error(test_prophet['y'].values, prophet_predictions),
                    'r2': r2_score(test_prophet['y'].values, prophet_predictions)
                }
            except Exception as e:
                print(f"\nWarning: Could not evaluate Prophet model: {e}")

        return results
    
    def visualize_results(self, results):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.ticker} Stock Price Prediction - Model Comparison', fontsize=16, fontweight='bold')

        # Plot 1: LSTM Predictions
        ax1 = axes[0, 0]
        ax1.plot(results['LSTM']['actual'], label='Actual Price', color='blue', linewidth=2)
        ax1.plot(results['LSTM']['predictions'], label='LSTM Predictions', color='red', linestyle='--', linewidth=2)
        ax1.set_title('LSTM Model Predictions', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Stock Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: ARIMA Predictions
        ax2 = axes[0, 1]
        ax2.plot(results['ARIMA']['actual'], label='Actual Price', color='blue', linewidth=2)
        ax2.plot(results['ARIMA']['predictions'], label='ARIMA Predictions', color='green', linestyle='--', linewidth=2)
        ax2.set_title('ARIMA Model Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Stock Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Prophet Predictions (if available)
        ax3 = axes[1, 0]
        if 'Prophet' in results:
            ax3.plot(results['Prophet']['actual'], label='Actual Price', color='blue', linewidth=2)
            ax3.plot(results['Prophet']['predictions'], label='Prophet Predictions', color='purple', linestyle='--', linewidth=2)
            ax3.set_title('Prophet Model Predictions', fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Prophet Model Not Available',
                    ha='center', va='center', fontsize=14, transform=ax3.transAxes)
            ax3.set_title('Prophet Model Predictions', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Stock Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance Comparison
        ax4 = axes[1, 1]
        models = [m for m in ['LSTM', 'ARIMA', 'Prophet'] if m in results]
        rmse_values = [results[m]['rmse'] for m in models]
        mae_values = [results[m]['mae'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, rmse_values, width, label='RMSE', color='coral')
        bars2 = ax4.bar(x + width/2, mae_values, width, label='MAE', color='lightblue')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Error Metrics')
        ax4.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('stock_predictions_comparison.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved!")
        
        return fig
    
    def plot_technical_indicators(self):
        """Visualize technical indicators"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(f'{self.ticker} Technical Indicators Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        ax1.plot(self.data.index, self.data['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_200'], label='SMA 200', alpha=0.7)
        ax1.set_title('Stock Price with Moving Averages')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # Plot 3: MACD
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', linewidth=2)
        ax3.plot(self.data.index, self.data['Signal_Line'], label='Signal Line', linewidth=2)
        ax3.bar(self.data.index, self.data['MACD_Histogram'], label='Histogram', alpha=0.3)
        ax3.set_title('MACD (Moving Average Convergence Divergence)')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume
        ax4 = axes[3]
        try:
            ax4.plot(self.data.index, self.data['Volume'], alpha=0.7, label='Volume', linewidth=1.5)
            ax4.plot(self.data.index, self.data['Volume_MA'], color='red', linewidth=2, label='Volume MA')
        except:
            # Fallback if index causes issues
            ax4.plot(self.data['Volume'].values, alpha=0.7, label='Volume', linewidth=1.5)
            ax4.plot(self.data['Volume_MA'].values, color='red', linewidth=2, label='Volume MA')
        ax4.set_title('Trading Volume')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volume')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('technical_indicators.png', dpi=300, bbox_inches='tight')
        print("Technical indicators visualization saved!")
        
        return fig
    
    def print_summary(self, results):
        """Print comprehensive model performance summary"""
        print("\n" + "="*80)
        print(f"MODEL PERFORMANCE SUMMARY - {self.ticker}")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name} Model:")
            print(f"  RMSE: ${metrics['rmse']:.2f}")
            print(f"  MAE:  ${metrics['mae']:.2f}")
            print(f"  R²:   {metrics['r2']:.4f}")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\n{'='*80}")
        print(f"Best Performing Model: {best_model[0]} (Lowest RMSE: ${best_model[1]['rmse']:.2f})")
        print(f"{'='*80}\n")


def main():
    """Main execution function"""
    
    # Configuration
    TICKER = 'AAPL'  # Apple Inc.
    START_DATE = '2020-01-01'
    END_DATE = '2024-10-30'
    
    print("="*80)
    print("STOCK MARKET PREDICTION SYSTEM")
    print("Models: LSTM + ARIMA + Prophet")
    print("="*80)
    
    # Initialize predictor
    predictor = StockPredictor(TICKER, START_DATE, END_DATE)
    
    # Fetch and process data
    predictor.fetch_data()
    predictor.calculate_technical_indicators()
    
    # Visualize technical indicators
    predictor.plot_technical_indicators()
    
    # Prepare data for LSTM
    X_train, X_test, y_train, y_test = predictor.prepare_lstm_data(
        lookback=60,
        features=['Close', 'SMA_20', 'RSI', 'MACD', 'Volume']
    )
    
    # Train models
    predictor.train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    predictor.train_arima(order=(5, 1, 2))

    # Try to train Prophet, but continue if it fails (common on Windows)
    try:
        predictor.train_prophet()
    except Exception as e:
        print(f"\nWarning: Prophet model failed to train: {e}")
        print("Continuing with LSTM and ARIMA models only...")
    
    # Evaluate and compare
    results = predictor.evaluate_models(X_test, y_test)
    predictor.visualize_results(results)
    predictor.print_summary(results)
    
    print("\nAnalysis complete! Check the output files for visualizations.")


if __name__ == "__main__":
    main()
