import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import TimeSeriesSplit #здесь проучим метод для кросс-валидации временных рядов

# Загрузка данных
df = pd.read_csv('../data/csv/main_data.csv')
#df['active-count'] = df['active-count_smoo']
df=df[df['date']>='2015-01-01']
#df=df[df['date']<='2022-01-01']

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ВАЖНО: Убедитесь что в данных нет будущей информации!
print("Проверка на утечку данных:")
print(f"Даты от {df['date'].min()} до {df['date'].max()}")

# Создание лаговых features (ПРАВИЛЬНО - только прошлые данные)
def create_lag_features(df, columns, n_lags=3):
    df = df.copy()
    for col in columns:
        for lag in range(1, n_lags+1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)  # shift(lag) - ПРОШЛЫЕ данные
    return df

# Целевая переменная - будущая цена (цена ЗАВТРА)
df['target_close'] = df['close'].shift(-1)

df['close_change'] = df['close'].pct_change(2)  # Изменение за 2 дня
df['close_change'] = df['close'].pct_change(3)  # Изменение за 3 дня

# Создание новых признаков
df['price_momentum_7d'] = df['close'].pct_change(periods=7)
df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
df['volume_ratio_7d'] = df['volume'] / df['volume_ma_7']

df['active_count_ma_7'] = df['active-count'].rolling(window=7).mean()

df['hash_active_count_7dirived'] = df['hash-rate']/df['active_count_ma_7']

df['returns'] = df['close'].pct_change()
df['volatility_14d'] = df['returns'].rolling(window=14).std()

df['rsi_divergence'] = df['rsi'] - df['close']

#df['market_cap_trend'] = df['marketCap'].pct_change(periods=7)

df['fee_to_volume_ratio'] = df['total_fee'] / df['volume']

df['macd_signal_diff'] = df['MACD'] - df['Signal_Line']

# Создание лаговых features (ТОЛЬКО на исторических данных)
features_to_lag = ['close_change', 'volume','rsi', 'MACD_Cross_Power_Normalized', 'hash-rate',
                   'active-count', 'total_fee', 'transfer_count', 'yuan', 'zew_state', 'zew_mood_index', 'rub_usd', 'gesi_value',
                   'price_momentum_7d', 'volume_ratio_7d', 'volatility_14d', 'rsi_divergence', 'fee_to_volume_ratio', 'macd_signal_diff'
    , 'hash_active_count_7dirived']
df = create_lag_features(df, features_to_lag, n_lags=7)

# Добавление скользящих средних (на исторических данных)
df['close_ma_7'] = df['close'].shift(1).rolling(7).mean()
df['volume_ma_7'] = df['volume'].rolling(window=7).mean()

# Добавьте больше лагов для mood index
df['zew_mood_ma_3'] = df['zew_mood_index'].shift(1).rolling(3).mean()
df['zew_mood_ma_7'] = df['zew_mood_index'].shift(1).rolling(7).mean()

# Скользящие средние для total_fee с правильным shift
df['total_fee_ma_3'] = df['total_fee'].shift(1).rolling(3).mean()
df['total_fee_ma_7'] = df['total_fee'].shift(1).rolling(7).mean()
df['total_fee_ma_14'] = df['total_fee'].shift(1).rolling(14).mean()

# Удаляем строки с NaN (первые строки после создания лагов и скользящих средних)
df = df.dropna()
# ВАЖНО: Убедимся что целевая переменная - это БУДУЩАЯ цена
print(f"Пример: date={df.iloc[0]['date']}, close={df.iloc[0]['close']}, target={df.iloc[0]['target_close']}")

selected_features = [
    # Только лаговые features и исторические данные!
    #'zew_mood_index_lag_1', 'zew_mood_index_lag_2', 'zew_mood_index_lag_3',
    #'active-count_lag_1', 'active-count_lag_2',
    #'active-count_lag_3', 'active-count_lag_4', 'active-count_lag_5',
    #'total_fee_lag_1', 'total_fee_lag_2', 'total_fee_lag_3',
    #'transfer_count_lag_1', 'transfer_count_lag_3',  'transfer_count_lag_5',
    #'close_change_lag_2', 'close_change_lag_3',
    #'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
    #'hash-rate_lag_1', 'hash-rate_lag_2', 'hash-rate_lag_3',
    #'rsi_lag_1', 'rsi_lag_2',
    #'MACD_Cross_Power_Normalized_lag_1', 'MACD_Cross_Power_Normalized_lag_2',
    #'rub_usd_lag_1', 'rub_usd_lag_2' , 'rub_usd_lag_3',
    #'gesi_value_lag_1', 'gesi_value_lag_2', 'gesi_value_lag_3',
    'close_ma_7',
    #'volume_ma_7',
    #'yuan_lag_1', 'yuan_lag_2', 'yuan_lag_3',
    #'zew_state_lag_1',
    #'zew_mood_ma_3',
    #'zew_mood_ma_7',
    #'total_fee_ma_3', 'total_fee_ma_7', 'total_fee_ma_14',
    'hash_active_count_7dirived',
    'price_momentum_7d',
    #'volume_ratio_7d',
    'volatility_14d',
    'rsi_divergence',
    'fee_to_volume_ratio',
    'macd_signal_diff'
]

df = df.dropna()

X = df[selected_features]
y = df['target_close']

# ХРОНОЛОГИЧЕСКОЕ разделение (без shuffle!)
train_size = int(len(df) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f"Train dates: {df.iloc[0]['date']} to {df.iloc[train_size-1]['date']}")
print(f"Test dates: {df.iloc[train_size]['date']} to {df.iloc[-1]['date']}")

# Масштабирование features (ТОЛЬКО на тренировочных данных)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ТЕСТ трансформируем параметрами от ТРЕЙНА


param_grid = {
    "n_estimators":[100, 150, 200, 300],
    "max_depth":[3, 5, 13, 15],
    "min_samples_split":[2, 5, 7, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
    "max_features": ['sqrt', 'log2', 0.5, 0.75],
    "random_state": [42],
    "subsample": [0.5, 0.7, 1.0],
    "min_samples_leaf": [1, 3, 10],
}

quick_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "max_features": ['sqrt', 'log2', 0.5, 0.75],
    "random_state": [42]
}

# тут я хочу добавить кросс-валидацию

tscv = TimeSeriesSplit(
    n_splits=20,          # у нас будет кроссвалидироваться временной ряд 20 раз (я брал из рассчета 2 раза в год на протяжении 10 лет)
    test_size=90,        # длина тестового набора - у нас полгода
    gap=7,               # буфер между train и test, спасает от look-ahead при лагах
)

# Обучение моделей
rf_model = RandomForestRegressor(
    # n_estimators=200
    # ,random_state=42
    # ,max_depth=5
    # ,min_samples_split=2
)

rf_model.fit(X_train_scaled, y_train)

gb_model = GradientBoostingRegressor(
    # n_estimators=200
    # ,random_state=42
    # ,max_depth=5
    # ,min_samples_split=2
    # ,learning_rate=0.5
    # ,criterion='squared_error'
    # pa
)

grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=quick_param_grid,
    cv=tscv,  # кросс-валидация
    scoring='neg_root_mean_squared_log_error',
    n_jobs=-1  # использование всех процессоров
)

grid_search.fit(X_train, y_train)

# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print("-" * 30)

    return predictions
def get_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importances.")

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    return feature_importance

# Оценка всех моделей
print("РЕАЛЬНЫЕ результаты:")
rf_pred = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
gb_pred = evaluate_model(grid_search, X_test_scaled, y_test, "Gradient Boosting")
grid_pred = evaluate_model(grid_search, X_test_scaled, y_test, "Grid")
# lr_pred = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
# Анализ важности признаков
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Важность признаков:")
print(feature_importance)
print("для grid search:")
print(get_importance(grid_search.best_estimator_, selected_features))

# Визуализация предсказаний
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Реальная цена', alpha=0.7)
# plt.plot(rf_pred, label='Random Forest', alpha=0.7)
plt.plot(gb_pred, label='Gradient Boosting', alpha=0.7)
plt.plot(grid_pred, label='Grid', alpha=0.7)
#plt.plot(lr_pred, label='Linear Regression', alpha=0.7)
plt.legend()
plt.title('Предсказания vs Реальная цена')
plt.savefig('predictions_V2.png')
plt.close()

print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая точность:", grid_search.best_score_)