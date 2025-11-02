# Stock Market Trend Prediction Using Machine Learning

## Overview
A comprehensive time-series forecasting system that combines **LSTM**, **ARIMA**, and **Prophet** models to predict stock market trends with enhanced accuracy through advanced feature engineering and hyperparameter optimization.

## 🎯 Key Features

### Advanced Modeling Techniques
- **LSTM (Long Short-Term Memory)**: Deep learning recurrent neural network optimized for sequential data
- **ARIMA (AutoRegressive Integrated Moving Average)**: Statistical time-series model for trend analysis
- **Prophet**: Facebook's forecasting tool designed for business time series

### Technical Indicators Integration
- **SMA (Simple Moving Averages)**: 20, 50, and 200-day moving averages
- **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and magnitude of price changes
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator

### Advanced Features
- Automated hyperparameter tuning
- Early stopping to prevent overfitting
- Multi-feature input for enhanced predictions
- Comprehensive performance metrics (RMSE, MAE, R²)
- Professional visualizations and comparative analysis

## 📊 Technical Implementation

### LSTM Architecture
```
Input Layer (60 timesteps × 5 features)
    ↓
LSTM Layer (128 units) + Dropout (0.2)
    ↓
LSTM Layer (64 units) + Dropout (0.2)
    ↓
LSTM Layer (32 units) + Dropout (0.2)
    ↓
Dense Output Layer (1 unit)
```

### Feature Engineering
- **Price Features**: Close, Open, High, Low
- **Moving Averages**: SMA 20/50/200
- **Momentum Indicators**: RSI, MACD, Signal Line
- **Volume Analysis**: Volume, Volume MA
- **Derived Features**: Daily returns, volatility

### Hyperparameter Optimization
- **LSTM**: 
  - Units: [128, 64, 32]
  - Dropout: 0.2
  - Learning Rate: 0.001
  - Batch Size: 32
  - Lookback Window: 60 days

- **ARIMA**: Order (5, 1, 2) - optimized through grid search
- **Prophet**: Daily, weekly, and yearly seasonality enabled

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
```

### Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage
```python
python stock_prediction.py
```

### Custom Configuration
```python
from stock_prediction import StockPredictor

# Initialize with custom parameters
predictor = StockPredictor(
    ticker='AAPL',
    start_date='2020-01-01',
    end_date='2024-10-30'
)

# Fetch and process data
predictor.fetch_data()
predictor.calculate_technical_indicators()

# Train models
X_train, X_test, y_train, y_test = predictor.prepare_lstm_data()
predictor.train_lstm(X_train, y_train, X_test, y_test)
predictor.train_arima()
predictor.train_prophet()

# Evaluate and visualize
results = predictor.evaluate_models(X_test, y_test)
predictor.visualize_results(results)
```

## 📈 Results & Performance

### Model Comparison
| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| LSTM | Optimized with deep architecture | Enhanced by technical indicators | High accuracy |
| ARIMA | Statistical baseline | Traditional time-series | Comparative benchmark |
| Prophet | Handles seasonality | Robust to missing data | Business-focused |

### Key Achievements
✅ **Improved Accuracy**: Advanced feature engineering with technical indicators  
✅ **Optimized Performance**: Hyperparameter tuning and early stopping  
✅ **Comprehensive Analysis**: Multi-model comparison framework  
✅ **Production-Ready**: Modular, scalable architecture  

## 📁 Project Structure
```
stock-market-prediction/
│
├── stock_prediction.py      # Main implementation
├── requirements.txt          # Project dependencies
├── README.md                # Documentation
│
└── outputs/
    ├── stock_predictions_comparison.png
    └── technical_indicators.png
```

## 🔧 Technical Stack

### Machine Learning & Deep Learning
- TensorFlow/Keras: LSTM implementation
- Statsmodels: ARIMA modeling
- Prophet: Time-series forecasting

### Data Processing
- Pandas: Data manipulation
- NumPy: Numerical computations
- Scikit-learn: Preprocessing and metrics

### Visualization
- Matplotlib: Static plots
- Seaborn: Statistical visualizations

### Financial Data
- yfinance: Real-time stock data API

## 📊 Output Visualizations

### 1. Model Predictions Comparison
- LSTM predictions vs actual prices
- ARIMA predictions vs actual prices
- Prophet predictions vs actual prices
- Performance metrics comparison

### 2. Technical Indicators Analysis
- Price with moving averages (SMA 20/50/200)
- RSI with overbought/oversold zones
- MACD with signal line and histogram
- Trading volume with moving average

## 🎓 Key Learnings

1. **Feature Engineering Impact**: Technical indicators significantly improve model performance
2. **Model Diversity**: Combining statistical and ML approaches provides robust predictions
3. **Hyperparameter Tuning**: Critical for achieving optimal LSTM performance
4. **Time-Series Challenges**: Handling non-stationarity and volatility in financial data

## 🔮 Future Enhancements

- [ ] Ensemble model combining all three approaches
- [ ] Real-time prediction API
- [ ] Sentiment analysis from financial news
- [ ] Multi-stock portfolio optimization
- [ ] Advanced technical indicators (Bollinger Bands, Fibonacci retracements)
- [ ] Attention mechanisms in LSTM
- [ ] Explainable AI for prediction interpretation

## 📝 Notes

- **Data Source**: Historical stock data from Yahoo Finance
- **Timeframe**: Configurable (default: 2020-2024)
- **Train/Test Split**: 80/20
- **Update Frequency**: Models can be retrained with latest data

## 🤝 Contributing

This project demonstrates advanced time-series forecasting techniques for portfolio purposes. Feel free to:
- Experiment with different stocks
- Try alternative model architectures
- Add new technical indicators
- Implement ensemble methods



---

## 🌟 Highlights

✨ **Advanced ML/DL**: LSTM neural networks with optimized architecture  
✨ **Financial Domain Knowledge**: Technical indicators (SMA, RSI, MACD)  
✨ **Model Comparison**: Multiple approaches with quantitative evaluation  
✨ **Production Code**: Clean, modular, well-documented implementation  
✨ **Data Visualization**: Professional charts and comparative analysis  
✨ **Hyperparameter Optimization**: Systematic tuning for best performance  

---

*Last Updated: October 2024*
