import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv('bitcoin_analysis/data/csv/main_data.csv')
df = df.drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)
df = df[df['date'] >= '2024-01-01'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Инженерия признаков
df['close_change'] = df['close'].pct_change(3)
df['hash_rate_ma_7'] = df['hash-rate'].rolling(7).mean()
df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
df['active_count_ma_7'] = df['active-count'].rolling(window=7).mean()
df['hash_active_count_7dirived'] = df['hash_rate_ma_7'] / (df['active_count_ma_7'] * df['volume'])
df['hash_active_count_dirived14'] = df['hash_active_count_7dirived'].rolling(window=14).mean()
df['macd_signal_diff'] = df['MACD'] - df['Signal_Line']
df['rsi_divergence'] = df['rsi'] - df['close']

features_to_lag = ['close', 'volume', 'rsi', 'MACD_Cross_Power_Normalized', 
                   'hash-rate', 'active-count', 'total_fee', 'transfer_count']

def create_lag_features(data, columns, n_lags=4):
    d = data.copy()
    for col in columns:
        for lag in range(1, n_lags+1):
            d[f'{col}_lag_{lag}'] = d[col].shift(lag)
    return d

df = create_lag_features(df, features_to_lag)
df['target_close'] = df['close'].shift(-1)
df = df.dropna()

selected_features = [
    'hash_active_count_dirived14', 'rsi', 'MACD_Cross_Power_Normalized', 
    'macd_signal_diff', 'rsi_divergence'
]
lag_cols = [col for col in df.columns if '_lag_' in col]
selected_features.extend(lag_cols)

X = df[selected_features]
y = df['target_close']

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
test_dates = df.iloc[train_size:]['date']

# 3. Метрики и Валидация
tscv = TimeSeriesSplit(n_splits=5, test_size=30, gap=7)

def NMAPE(y_true, y_pred):
    return 1 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100

scoring = {'r2': 'r2', 'mape': make_scorer(NMAPE), 'mse': 'neg_mean_squared_error'}

# 4. Пайплайны
gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(random_state=42))
])

elastic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ElasticNet(random_state=42, max_iter=50000, tol=1e-2))
])

# 5. Сетки параметров
gb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1]
}

ridge_param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

elastic_param_grid = {
    'model__alpha': [10.0, 100.0, 500.0, 1000.0],
    'model__l1_ratio': [0.5, 0.7, 0.9, 0.99]
}

# 6. Обучение всех моделей
models = {
    "Gradient Boosting": (gb_pipeline, gb_param_grid),
    "Ridge": (ridge_pipeline, ridge_param_grid),
    "Elastic Net": (elastic_pipeline, elastic_param_grid)
}

results = {}

for name, (pipeline, grid) in models.items():
    print(f"Обучение {name}...")
    gs = GridSearchCV(estimator=pipeline, param_grid=grid, cv=tscv, scoring=scoring, refit='r2', n_jobs=-1)
    gs.fit(X_train, y_train)
    
    preds = gs.predict(X_test)
    results[name] = {
        'predictions': preds,
        'r2': r2_score(y_test, preds),
        'mse': mean_squared_error(y_test, preds),
        'mape': NMAPE(y_test.values, preds),
        'params': gs.best_params_
    }

# 7. Сравнение результатов
print("\n" + "="*40)
print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*40)
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MSE:   {metrics['mse']:.0f}")
    print(f"  NMAPE: {metrics['mape']:.4f}")
    print(f"  Параметры: {metrics['params']}\n")

# 8. Визуализация
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test.values, label='Реальная цена', color='black', linewidth=2)

colors = {'Gradient Boosting': 'red', 'Ridge': 'green', 'Elastic Net': 'blue'}
for name, metrics in results.items():
    plt.plot(test_dates, metrics['predictions'], label=f"{name} (R²={metrics['r2']:.2f})", color=colors[name], alpha=0.7)

plt.xlabel('Дата')
plt.ylabel('Цена BTC')
plt.title('Битва алгоритмов: Gradient Boosting vs Ridge vs Elastic Net')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models_comparison.png')
print("График сравнения сохранен как 'models_comparison.png'")