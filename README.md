# Ongole AQI Predictor

Machine learning model to predict the Air Quality Index (AQI) in Ongole, Andhra Pradesh.

## Overview
Predicts future AQI levels in Ongole using historical air quality data, weather variables, and time-series modeling. Useful for local planning and health awareness around RGUKT Ongole and the region.

## Features
- Time-series forecasting (LSTM, ARIMA, or Prophet)
- Feature engineering with weather data (temperature, humidity, wind)
- Model evaluation metrics (RMSE, MAE, R²)
- Prediction visualizations and confidence intervals
- Support for daily/hourly forecasts

## Tech Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras or Statsmodels, Matplotlib/Seaborn
- **Data**: Historical AQI + meteorological data for Ongole

## Methodology
1. Data collection and cleaning
2. Exploratory data analysis
3. Feature selection and engineering
4. Model training and hyperparameter tuning
5. Evaluation and deployment-ready predictions

## Getting Started
```bash
git clone https://github.com/chan424/ongole-aqi-predictor.git
cd ongole-aqi-predictor
pip install -r requirements.txt
jupyter notebook  # or python predict.py
```

## Data
Place historical AQI CSV files in the `data/` directory.

## License
MIT
