#попробую воспроизвести из нашей каши
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.model_selection import TimeSeriesSplit #здесь проучим метод для кросс-валидации временных рядов

#щас загружу данные
df = pd.read_csv('../data/csv/main_data.csv')
df = df.drop_duplicates(subset=['date'], keep='first')
df = df.reset_index(drop=True)

df=df[df['date']>='2024-01-01']

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

#тут мы генерируем признаки - много кода с пандасовскими функциями
df['close_change'] = df['close'].pct_change(3)  # ПРОЦЕНТНОЕ Изменение за 3 дня

# Создание новых признаков
df['hash_rate_ma_7'] = df['hash-rate'].rolling(7).mean()
df['close_ma_7'] = df['close'].shift(1).rolling(7, min_periods=1).mean() #с какой целью здесь shift(1) - чтобы модель не подглядывала
df['volume_ma_7'] = df['volume'].rolling(window=7).mean() #скользящее среднее по объему

df['price_momentum_7d'] = df['close'].pct_change(periods=7) #процентное изменение за 7 дней
df['volume_ratio_7d'] = df['volume'] / df['volume_ma_7'] #по приколу ввели

df['active_count_ma_7'] = df['active-count'].rolling(window=7).mean() #то же самое скользящее среднее

df['hash_active_count_7dirived'] = df['hash-rate']/df['active_count_ma_7'] #как же чувствуем - признк пальцем в небо (почти)

df['returns'] = df['close'].pct_change() # ---||---
df['volatility_14d'] = df['returns'].rolling(window=14).std() #скользящее стандартное отклонения

df['rsi_divergence'] = (df['rsi'] - df['close']) #добавил
#не просто разность, а процентное изменение

df['fee_to_volume_ratio'] = df['total_fee'] / df['volume'] #все понято

df['macd_signal_diff'] = df['MACD'] - df['Signal_Line'] #когда MACD пересекает сигнал - это мощный намек

df['zew_mood_ma_3'] = df['zew_mood_index'].shift(1).rolling(3).mean() #по сути это тоже лаги, но со скользищим средним
df['zew_mood_ma_7'] = df['zew_mood_index'].shift(1).rolling(7).mean() # ---||---

df['total_fee_ma_3'] = df['total_fee'].shift(1).rolling(3).mean() #и снова тут лаги со скользящими средними
df['total_fee_ma_7'] = df['total_fee'].shift(1).rolling(7).mean() #---||---
df['total_fee_ma_14'] = df['total_fee'].shift(1).rolling(14).mean() #---||---


#!!!!
#тут забор
##############################################################################################################################

#######     ####    #    #   #   #   #####     ####        #       #       #   #   #    #   #   #        #
#     #    #    #   #   ##    # #    #        #    #      # #      #       #   #   #   ##   #   #       # #
#     #    #        #  # #     #     ####     #    #     #   #     ####    #####   #  # #   #   #      #####
#     #    #    #   # #  #    # #    #   #    #    #    #     #    #   #   #   #   # #  #   #######   #     #
#     #     ####    #    #   #   #   #####     ####    #       #   #####   #   #   #    #         #  #       #

df['hash_active_count_7dirived'] = df['hash_rate_ma_7']/(df['active_count_ma_7'] *df['volume'])
#эту дичь сделал Макар
df['hash_active_count_dirived14'] = df['hash_active_count_7dirived'].rolling(window=14).mean()

##############################################################################################################################
#тут кончился

#дальше здесь добавление лаговых features - это просто перекопирую
features_to_lag = ['close', 'close_change', 'volume','rsi', 'MACD_Cross_Power_Normalized', 'hash-rate',
                   'active-count', 'total_fee', 'transfer_count', 'zew_state', 'zew_mood_index', 'gesi_value'
                   ]
#дальше тут функция, котороая будет добавльть в наш датасет лаговые фичи - СРАЗУ СО СМЕЩЕНИЕМ (будем использовать shift)
def create_lag_features(df, columns, n_lags=4):
    df = df.copy()
    for col in columns:
        for lag in range(1, n_lags+1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)  # shift(lag) - ПРОШЛЫЕ данные
    return df
df = create_lag_features(df, features_to_lag)
#в результет мы получим новые столбцы со смещением в наш датасет

#добавим булевый столбец - цена повысилась или упала
df['price_change'] = (df['close'] > df['close'].shift(1))

#сейчас определим целевую переменную - она потом будет в Y наборе данных
df['target_close'] = df['close'].shift(-1)

# Удаляем строки с NaN (первые строки после создания лагов и скользящих средних)
df = df.dropna()

#клево, создали много интересных фичей, теперь для удобства их можно объединить в массив

selected_features = [
    'close_lag_4',
    #'zew_mood_index_lag_1', 'zew_mood_index_lag_2', 'zew_mood_index_lag_3',
    'active-count_lag_1', 'active-count_lag_2',
    #'active-count_lag_3', 'active-count_lag_4', 'active-count_lag_5',
    'hash_active_count_dirived14',
    'total_fee_lag_1', 'total_fee_lag_2', 'total_fee_lag_3',
    'transfer_count_lag_1', 'transfer_count_lag_3',
    'close_change_lag_2', 'close_change_lag_3',
    #'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
    'hash-rate_lag_1', 'hash-rate_lag_2', 'hash-rate_lag_3',
    'rsi_lag_1', 'rsi_lag_2',
    'MACD_Cross_Power_Normalized_lag_1', 'MACD_Cross_Power_Normalized_lag_2',
    #'rub_usd_lag_1', 'rub_usd_lag_2' , 'rub_usd_lag_3',
    #'gesi_value_lag_1', 'gesi_value_lag_2', 'gesi_value_lag_3',
    'close_ma_7',
    'volume_ma_7',
    #'yuan_lag_1', 'yuan_lag_2', 'yuan_lag_3',
    #'zew_state_lag_1',
    #'zew_mood_ma_3',
    #'zew_mood_ma_7',
    'total_fee_ma_3', #'total_fee_ma_7', 'total_fee_ma_14',
    'hash_active_count_7dirived',
    'price_momentum_7d',
    'volume_ratio_7d',
    'volatility_14d',
    'rsi_divergence',
    'fee_to_volume_ratio',
    'macd_signal_diff'
]

#мы закончили готовить фичи, теперь явно разделим датасет на X и Y
#надо потом не забыть про нормализацию(масштабирование)

# Детальная проверка на утечку данных
print("=== ПРОВЕРКА НА УТЕЧКИ ДАННЫХ ===")

# 1. Проверка: есть ли в фичах информация из будущего?
for col in df.columns:
    # Ищем корреляцию с будущими значениями
    corr_with_future = df[col].corr(df['close'].shift(-1))
    if abs(corr_with_future) > 0.8:
        print(f"⚠️ ВОЗМОЖНА УТЕЧКА: {col} имеет высокую корреляцию с будущей ценой: {corr_with_future:.3f}")

# 2. Проверка временных меток
print(f"\nДиапазон дат в фичах: {df['date'].min()} - {df['date'].max()}")
print(f"Диапазон дат в target: {df[df['target_close'].notna()]['date'].min()} - {df[df['target_close'].notna()]['date'].max()}")

# 3. Проверка конкретных подозрительных фич
suspect_features = ['close_ma_7', 'hash_active_count_dirived14', 'hash_active_count_7dirived']
for feature in suspect_features:
    if feature in df.columns:
        # Проверяем, не использует ли фича будущие данные
        temp_df = df[['date', 'close', feature]].copy()
        temp_df['future_close'] = temp_df['close'].shift(-1)
        corr = temp_df[feature].corr(temp_df['future_close'])
        print(f"Корреляция {feature} с будущей ценой: {corr:.3f}")

y = df['target_close']
X = df[selected_features]

#теперь разделим на train и test данные

#концепция следующая: на тренировочных данных модель при
# обучении имеет полный доступ к (целевым данным в y) и подстраивает свои
# данные из X(выбранные фичи) под ответы из y - учится, развивается, находит
# закономерности
# а потом на тестовых данных модель будет использовать эти закономерности и пытаться
# угадать ответы из y - при этом здесь уже она не знает данных из y
# "Модель НЕ обучается и НЕ меняется на тестовых данных" - это важно
# ниже я это и сделаю, но при применении кросс-валидации на временных рядах мы к этому
# еще вернемся !!! - там важное уточнение

train_size = int(len(df) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

test_dates = df.iloc[train_size:]['date']

print("Тестовые данные ")

#далее настройка кросс-валидации для временных рядов - концепция в том, чтобы
#разделить тренировочный промежуток на несколько частей и сначала на первой части
#разделить данные на train и test, а при переходе на вторую часть train2 = train1 + test1
#и так далее
#ВАЖНО! хотя мы тут берем тестовые и тренировочные данные по разному, но тестовые данные, которые
#мы указали выше все равно не будут затронуты в процессе кросс-валидации - и вообще модель
#НИКОГДА не будет на них учиться (в нашем случае - на последних 20% данных)

tscv = TimeSeriesSplit(
    n_splits=5,          # у нас будет кроссвалидироваться временной ряд 20 раз (я брал из рассчета 2 раза в год на протяжении 10 лет)
    test_size=30,        # длина тестового набора - у нас полгода
    gap=7,               # буфер между train и test, спасает от look-ahead при лагах
)#еще будем настраивать

#злесь будет скоринг - набор оценочных метрик(из психбольницы)
def NMAPE(y_true, y_pred):
    return 1 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def directional_accuracy(y_true, y_pred):
    """Точность предсказания направления изменения"""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)


def symmetric_mape(y_true, y_pred):
    """SMAPE метрика (более сбалансированная)"""
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


#tscv = TimeSeriesSplit(n_splits=5, test_size=30, gap=0)
refit = 'r2'
scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mape': make_scorer(NMAPE),
    'mse': 'neg_mean_squared_error',
    'directional_accuracy': make_scorer(directional_accuracy),
    'symmetric_mape': make_scorer(symmetric_mape),
}


#тепрь для каждой модели пропишем пайплайн - удобная шняга, с ней у нас точно не утекут данные
#и выглядит это лучше
#"Это инструмент, который последовательно применяет список преобразований к данным
# и в конце применяет модель. Все шаги объединяются в единый объект."
# Пайплайн для Gradient Boosting

gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Этап 1: масштабирование
    ('model', GradientBoostingRegressor(random_state=42))  # Этап 2: модель
])

# Пайплайн для Ridge
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),#масштабирование данных - мы приводим цифры к одному масштабу
    ('model', Ridge(random_state=42))
])
#дальше настрою сетки - для градиентного бустинга и риджа

gb_param_grid = {
    "model__n_estimators":[100, 150, 200, 300], #количество деревьев
    "model__max_depth":[3, 5, 13, 15],
    "model__min_samples_split":[2, 5, 7, 10],#минимальное количество объектов для разбиения
    'model__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
    "model__max_features": ['sqrt', 'log2', 0.5, 0.75], #предельно понятно
    "model__random_state": [42],#это просто сид для генератора
    "model__subsample": [0.5, 0.7, 1.0],#процент обучающих данных
    "model__min_samples_leaf": [1, 3, 10],#минимальное количество объектов в листе
}

ridge_param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__random_state': [42]
}#потом дополним если нужно будет

quick_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__max_features": ['sqrt', 'log2', 0.5, 0.75],
    "model__random_state": [42]
}

#model__ пищем потому что используем с пайплайном

#теперь на основе сеток, пайплайнов и кросс-валидации настроим градиентный бустинг

gb_grid_search = GridSearchCV(
    estimator=gb_pipeline, #тут еще можно вместо пайплайна просто имя модели
    param_grid=quick_param_grid,
    cv=tscv,  # кросс-валидация которую я выше настроил
    scoring=scoring,#метрика для оценки
    n_jobs=8, # тут интереснее - это для параллельного выполнения (-1 - все ядра)
    verbose=3, #насколько подробно будет выводиться информация о процессе
    refit=refit #тут тоже интересно - если True, то мы сразу переобучим модель на лучших данных
    #если False, то нам это нужно будет делать вручную
)

ridge_grid_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=ridge_param_grid,
    cv=tscv,
    scoring=scoring,
    n_jobs=8,
    verbose=3,
    refit=refit
)

print("Обучение Gradient Boosting...")
gb_grid_search.fit(X_train, y_train)  # Автоматически делает кросс-валидацию

print("Обучение Ridge...")
ridge_grid_search.fit(X_train, y_train) # метод fit - для ОБУЧЕНИЯ модели на ТРЕНИРОВОЧНЫХ данных
#после двух фитов мы получили лучшие модели

best_gb = gb_grid_search.best_estimator_
best_ridge = ridge_grid_search.best_estimator_
#у нас grid_search перебрала все гиперпараметры, нашла лучшие и (потому что refit=True) обучила
# обучила модель на лучших гиперпараметрах и записала ее в свой аттрибут best_estimator_
# вообще говря, мы в отдельную модель записали для удобства, так то она и так хранится в
# gb_grid_search

#супер, у нас есть обученные модели, время проверить их на тестовых данных и вывести результаты

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print("-" * 30)

    return predictions
#настал самый волнительный момент в жизни наших моделей - чуваки учились учились и тепрь прогнозируют
gb_pred = evaluate_model(best_gb, X_test, y_test, "Gradient Boosting")
ridge_pred = evaluate_model(best_ridge, X_test, y_test, "Ridge")

def get_importance(model, features): #очень полезная шняга, которая позволяет узнать важность фичей
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

#важность фичей для градиентного бустинга

feature_importanceGB = pd.DataFrame({
    'feature': selected_features,
    'importance': best_gb.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-5 важных признаков GBoost:")
print(feature_importanceGB.head(5))

feature_importanceRidge = pd.DataFrame({
    'feature': selected_features,
    'importance': best_ridge.named_steps['model'].coef_ #best_ridge.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-5 важных признаков Ridge:")
print(feature_importanceRidge.head(5))

# Добавьте это после создания baseline_pred
# window_size = 20  # Размер окна сглаживания
# sma_baseline = y_test.rolling(window=window_size).mean().dropna()

# Визуализация предсказаний
plt.figure(figsize=(12, 6))
window_size = 15
sma_baseline = y_test.rolling(window=window_size).mean()
aligned_test_dates = test_dates[:len(sma_baseline)]

# Проверяем, чтобы не было NaN
sma_values = sma_baseline.dropna()
aligned_test_dates = aligned_test_dates[len(sma_baseline) - len(sma_values):]

plt.plot(test_dates,y_test.values, label='Реальная цена', alpha=0.7)
# plt.plot(rf_pred, label='Random Forest', alpha=0.7)
plt.plot(test_dates,gb_pred, label='Gradient Boosting', alpha=0.7)
plt.plot(test_dates,ridge_pred, label='Ridge', alpha=0.7)
#plt.plot(test_dates[:len(sma_baseline)],sma_baseline.values, label='Baseline', alpha=0.7)
plt.plot(aligned_test_dates, sma_values.values, label='Baseline (SMA)', alpha=0.7, linestyle='--')

plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.grid()
plt.tight_layout()
plt.title('Предсказания vs Реальная цена')
plt.savefig('predictions_V2.png')
plt.xticks(rotation=90)
plt.show()

print("Лучшие параметры для GB:", gb_grid_search.best_params_)
print("Лучшая точность GB:", gb_grid_search.best_score_)

print("Лучшие параметры для Ridge:", ridge_grid_search.best_params_)
print("Лучшая точность Ridge:", ridge_grid_search.best_score_)