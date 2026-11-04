import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data generation (replace with real Ongole AQI data)
def generate_sample_data(n=1000):
    np.random.seed(42)
    data = {
        'temp': np.random.normal(30, 5, n),
        'humidity': np.random.normal(60, 15, n),
        'wind_speed': np.random.normal(5, 2, n),
        'aqi': np.random.normal(80, 30, n)
    }
    return pd.DataFrame(data)

def train_model():
    df = generate_sample_data()
    X = df[['temp', 'humidity', 'wind_speed']]
    y = df['aqi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: {rmse:.2f}")
    return model

if __name__ == "__main__":
    model = train_model()
    print("Sample prediction for temp=32, humidity=55, wind=4:", model.predict([[32, 55, 4]])[0])
