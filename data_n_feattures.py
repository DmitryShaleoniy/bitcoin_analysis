# Загрузка данных
import pandas as pd
import matplotlib.pyplot as plt
import cv

df = pd.read_csv('main_data.csv')
#df['active-count'] = df['active-count_smoo']
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
df['close_ma_3'] = df['close'].shift(1).rolling(3).mean()

df['close_change'] = df['close'].pct_change(2)  # Изменение за 2 дня
df['close_change'] = df['close'].pct_change(3)  # Изменение за 3 дня

# Создание новых признаков
df['price_momentum_7d'] = df['close'].pct_change(periods=7)
df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
df['volume_ratio_7d'] = df['volume'] / df['volume_ma_7']

df['active_count_ma_7'] = df['active-count'].rolling(window=7).mean()
df['hash_rate_ma_7'] = df['hash-rate'].rolling(window=7).mean()

df['hash_active_count_7dirived'] = df['hash_rate_ma_7']/(df['active_count_ma_7'] *df['volume'])
df['hash_active_count_dirived14'] = df['hash_active_count_7dirived'].rolling(window=14).mean()

df['returns'] = df['close'].pct_change()
df['volatility_14d'] = df['returns'].rolling(window=14).std()

df['rsi_divergence'] = df['rsi'] - df['close']

#df['market_cap_trend'] = df['marketCap'].pct_change(periods=7)

df['fee_to_volume_ratio'] = df['total_fee'] / df['volume']

df['macd_signal_diff'] = df['MACD'] - df['Signal_Line']


# Создание лаговых features (ТОЛЬКО на исторических данных)
features_to_lag = [
    'close_change'
    , 'volume'
    ,'rsi'
    , 'MACD_Cross_Power_Normalized'
    , 'hash-rate'
    ,'active-count'
    , 'total_fee'
    , 'transfer_count'
    # , 'yuan'
    , 'zew_state'
    , 'zew_mood_index'
    # , 'rub_usd'
    , 'gesi_value'
    , 'price_momentum_7d'
    , 'volume_ratio_7d'
    , 'volatility_14d'
    , 'rsi_divergence'
    , 'fee_to_volume_ratio'
    , 'macd_signal_diff'
    , 'hash_active_count_7dirived'
    , 'hash_active_count_dirived14'
]
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


plt.figure(figsize=(15, 10))
plt.plot(df['date'], df['close'], label='Close Price', color='green')
plt.plot(df['date'], df['hash_active_count_dirived14'], label='hash_active_count_dirived14')
plt.plot(df['date'], df['MACD'], label='MACD', color='orange')
plt.plot(df['date'], df['volume_ma_7']/ 1e7, label='volume_ma_7')
# plt.plot(df['date'], df['rsi_divergence'], label='rsi_divergence', color='blue')
plt.plot(df['date'], df['active_count_ma_7'] / 20, label='active_count_ma_7', color='red')
# plt.plot(df['date'], df['transfer_count'] / 3e1, label='transfer_count', color='blue')
plt.plot(df['date'], df['rsi'] * 1e2*2, label='rsi',)
plt.legend()
plt.savefig('close_price.png', dpi=300)
plt.close()

print(df['hash_active_count_7dirived'].tail())

df.to_csv('features.csv', index=False)
