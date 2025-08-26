import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler

# from cv import directional_accuracy, NMAPE, symmetric_mape


def NMAPE(y_true, y_pred):
    return 1 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def directional_accuracy(y_true, y_pred):
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)


def symmetric_mape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100




model = joblib.load('ridge_model.pkl')
data = pd.read_csv("features.csv")
data['date'] = pd.to_datetime(data['date'])
data = data[ data['date'] >= dt.datetime(2017,1 ,1) ]
data = data[ data['date'] <= dt.datetime(2020, 1, 1) ]
data = data.reset_index(drop=True)
data = data.dropna()

target = 'close'
features = [
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
    # 'close_ma_7',
    # 'close_ma_3',
    # 'close_lag_1',
    'hash-rate_lag_1', 'hash-rate_lag_2',
    'hash_active_count_dirived14',
    'hash_active_count_dirived14_lag_1', 'hash_active_count_dirived14_lag_2', 'hash_active_count_dirived14_lag_3',
    # 'rsi_lag_1',
    # 'rsi_lag_2',
    # 'rs',
    'volatility_14d',
    'rsi_divergence',
    'macd_signal_diff',
    'fee_to_volume_ratio',
    'MACD_Cross_Power_Normalized_lag_1',
    # Добавьте макроэкономические показатели с лагами
    # 'yuan_lag_1', 'rub_usd_lag_1',
    'gesi_value_lag_1'
]

X = data[features]
y = data[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pred = model.predict(X_scaled)
plt.figure(figsize=(10, 6))
plt.plot(y, label='Actual',
         color='green', alpha = 0.5,
         linewidth=3)
plt.plot(pred, label='Predicted',
         color='black', alpha = 1)
plt.savefig('validating_model_all_data.png')

