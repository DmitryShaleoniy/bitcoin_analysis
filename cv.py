#df=df[df['date']>='2023-01-01']
# from lib2to3.fixer_util import make_suite

# from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer



df = pd.read_csv('features.csv')
df['gain_or_loss'] = (df['close'] - df['close'].shift(1)) > 0
df['date'] = pd.to_datetime(df['date'])
df=df[df['date'] >= '2020-01-01']
print(df[['date', 'close', 'gain_or_loss']].tail())
df = df.dropna()



all_features = [
    # Только лаговые features и исторические данные!
    #'zew_mood_index_lag_1', 'zew_mood_index_lag_2', 'zew_mood_index_lag_3',
    # 'active-count_lag_1', 'active-count_lag_2',
    #'active-count_lag_3', 'active-count_lag_4', 'active-count_lag_5',
    'total_fee_lag_1',
    # 'total_fee_lag_2', 'total_fee_lag_3',
    #'transfer_count_lag_1', 'transfer_count_lag_3',  'transfer_count_lag_5',
    # 'close_change_lag_2', 'close_change_lag_3',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
    'hash-rate_lag_1',
    'hash-rate_lag_2', 'hash-rate_lag_3',
    #'rsi_lag_1', 'rsi_lag_2',
    'MACD_Cross_Power_Normalized_lag_1', 'MACD_Cross_Power_Normalized_lag_2',
    #'rub_usd_lag_1', 'rub_usd_lag_2' , 'rub_usd_lag_3',
    # 'gesi_value_lag_1', 'gesi_value_lag_2', 'gesi_value_lag_3',
    # 'close_ma_7',
    'volume_ma_7',
    #'yuan_lag_1', 'yuan_lag_2', 'yuan_lag_3',
    #'zew_state_lag_1',
    #'zew_mood_ma_3',
    #'zew_mood_ma_7',
    #'total_fee_ma_3', 'total_fee_ma_7', 'total_fee_ma_14',
    # 'hash_active_count_7dirived',
    # 'price_momentum_7d',
    'hash_active_count_dirived14_lag_1', 'hash_active_count_dirived14_lag_2', 'hash_active_count_dirived14_lag_3',
    'hash_active_count_dirived14',
    'volume_ratio_7d',
    'volatility_14d',
    # 'rsi_divergence',
    'fee_to_volume_ratio',
    'macd_signal_diff'
]

good_features = [
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

target = 'close'

X = df[good_features]
y = df[target]

# ХРОНОЛОГИЧЕСКОЕ разделение (без shuffle!)
train_size = int(len(df) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f"Train dates: {df.iloc[0]['date']} to {df.iloc[train_size-1]['date']}")
print(f"Test dates: {df.iloc[train_size]['date']} to {df.iloc[-1]['date']}")
train_dates = df.iloc[:train_size]['date']
test_dates = df.iloc[train_size:]['date']

############################################################################# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############################################################################## Гиперпараметры

param_grid = {
    "n_estimators":[20, 50, 70, 80, 100],
     "max_depth":[1, 2, 3],
    "min_samples_split":[2, 5, 10, 15, 20, 30],
    'learning_rate': [0.1, 0.05],
    "max_features": [1.0],
    "random_state": [42],
    "subsample": [1.0],
}

ridge_param_grid = {
    'alpha': [10, 5, 2, 1.7, ], #1.5, 1.2, 1, 0.5], # , 0.01, 0.001],
    'fit_intercept': [True],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    'max_iter': [10000],
    'tol': [1e-6]

}

############################################################# Скоринг и глобальные переменные

tscv = TimeSeriesSplit(n_splits=5, test_size=30, gap=7)
refit ='symmetric_mape'

def NMAPE(y_true, y_pred):
    return 1 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def directional_accuracy(y_true, y_pred):
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)


def symmetric_mape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    # 'mse': 'neg_mean_squared_error',
    # 'mape': make_scorer(NMAPE),
    'directional_accuracy': make_scorer(directional_accuracy),
    'symmetric_mape': make_scorer(symmetric_mape),
}

####################################################################### Модели и обучение
gb_model = GradientBoostingRegressor()
ridge_model = Ridge()

# {'alpha': 1, 'fit_intercept': True, 'max_iter': 10000, 'solver': 'sparse_cg', 'tol': 1e-06}
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring=scoring,
    refit=refit,
    n_jobs=8
)
ridge_grid_search = GridSearchCV(
    estimator=ridge_model,
    param_grid=ridge_param_grid,
    cv=tscv,
    scoring=scoring,
    refit=refit,
    n_jobs=8
)
grid_search.fit(X_train_scaled, y_train)
ridge_grid_search.fit(X_train_scaled, y_train)

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = NMAPE(y_test, predictions)
    Symmetric_mape = symmetric_mape(y_test, predictions)

    print(f"{model_name}:")
    print(f"symmetric_mape: {Symmetric_mape:.4f}")
    print(f"mape: {mape:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print("-" * 30)

    return predictions

########################################################################### Выводы и Оценка
print("Результаты:")
print("-" * 30)

gb_pred = evaluate_model(grid_search, X_test_scaled, y_test, "Gradient Boosting")
ridge_pred = evaluate_model(ridge_grid_search, X_test_scaled, y_test, "Ridge")

# Анализ важности признаков
def get_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Убираем [0], потому что coef_ уже содержит 1D-массив
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature importances.")

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values(by='importance', ascending=False)

    return feature_importance

print("Важность признаков Boosting:")
print(get_importance(grid_search.best_estimator_, good_features))
print("Важность признаков Ridge:")
print(get_importance(ridge_grid_search.best_estimator_, good_features))

###################################################################### Визуализация
plt.figure(figsize=(12, 6))
window_size = 15
sma_baseline = y_test.rolling(window=window_size).mean().dropna()

plt.plot(y_test.values,
         label='Реальная цена',
         alpha=0.3,
         linewidth=5,
         # linestile='dashed',
         color='red'
         )
plt.plot(range(window_size-1, len(y_test)), sma_baseline.values,
         label=f'Baseline (SMA {window_size} дней)', alpha=0.7, linewidth=2, color='purple')
plt.plot(gb_pred,
         label='Gradient Boosting',
         alpha=1,
         color='blue')
plt.plot(ridge_pred,
         label='Ridge',
         alpha=1,
         color = 'green')


plt.title("Сравнение прогноза и реальных данных")
plt.xlabel("Дата")
plt.ylabel("Цена закрытия")

plt.legend()
plt.grid(True)
plt.xticks( np.arange(0, len(test_dates), step=len(test_dates)//10),
            rotation=90)
plt.tight_layout()

plt.savefig('predictions.png')

#################################################################### Дополнительно
print(grid_search.best_params_)
print(ridge_grid_search.best_params_)
print(len(test_dates), len(gb_pred), len(ridge_pred))


import joblib
joblib.dump(grid_search, 'gb_model.pkl')
joblib.dump(ridge_grid_search, 'ridge_model.pkl')