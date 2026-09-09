import os
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. ПРОВЕРКА АКТУАЛЬНОСТИ ДАННЫХ И ПАРСИНГ
# ==========================================

main_data_path = './data/csv/main_data.csv'
need_update = True

if os.path.exists(main_data_path):
    try:
        df_main = pd.read_csv(main_data_path)
        last_date = str(df_main['date'].max())
        yesterday_str = (dt.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Если последняя дата >= вчерашней, парсер можно не запускать
        if last_date >= yesterday_str:
            need_update = False
    except Exception as e:
        print(f"Ошибка чтения main_data.csv: {e}. Требуется обновление.")

if need_update:
    print("Данные устарели или отсутствуют. Запускаем парсер...")
    os.system("python ./parser/parser_test.py")
else:
    print("Данные актуальны. Переходим к агрегации.")


# ==========================================
# 2. ЗАГРУЗКА И РАСЧЕТ ИНДИКАТОРОВ (BTC)
# ==========================================

# Основной датафрейм (цены Биткоина)
df = pd.read_csv('./data/csv/btc_no_vibrosi_copy.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

df['change'] = df['close'] - df['open']
df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
df['loss'] = df['change'].apply(lambda x: -x if x < 0 else 0)

df['gain_avg_14'] = df['gain'].rolling(14).mean()
df['loss_avg_14'] = df['loss'].rolling(14).mean()

# Защита от деления на 0 при расчете RSI
df['rs'] = (df['gain_avg_14'] / df['loss_avg_14']).replace([np.inf, -np.inf], np.nan).fillna(0)
df['rsi'] = (100 - (100 / (1 + df['rs']))).round(2)

df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

df['MACD_Cross_Power_Normalized'] = df['MACD_Histogram'] / df['close']


# ==========================================
# 3. ПОДКЛЮЧЕНИЕ МЕТРИК ИЗ CSV
# ==========================================

# 1. Средний размер блока (bsize)
block_df = pd.read_csv('./data/csv/avg_size.csv')
block_df['date'] = pd.to_datetime(block_df['date'])
block_df = block_df.rename(columns={'value': 'bsize'})

# 2. Хешрейт
hash_df = pd.read_csv('./data/csv/btc_hash_rate.csv')
hash_df['date'] = pd.to_datetime(hash_df['date'])
hash_df = hash_df.rename(columns={'value': 'hash-rate'})

# 3. Активные адреса
active_count_df = pd.read_csv('./data/csv/btc_active_addresses.csv')
active_count_df['date'] = pd.to_datetime(active_count_df['date'])
active_count_df = active_count_df.rename(columns={'value': 'active-count'})

# 4. Суммарный объем комиссий/транзакций (total_fee)
volume_sum_df = pd.read_csv('./data/csv/volume_sum.csv')
volume_sum_df['date'] = pd.to_datetime(volume_sum_df['date'])
volume_col = 'volume_sum' if 'volume_sum' in volume_sum_df.columns else 'value'
volume_sum_df = volume_sum_df.rename(columns={volume_col: 'total_fee'})

# 5. Суточный объем переводов
transfers_df = pd.read_csv('./data/csv/transfers_volume_sum.csv')
transfers_df['date'] = pd.to_datetime(transfers_df['date'])
transfers_col = 'value_usd' if 'value_usd' in transfers_df.columns else 'value'
transfers_df = transfers_df.rename(columns={transfers_col: 'transfer_count'})

# 6. Макроэкономика (GESI)
if os.path.exists('./data/csv/gesi.csv'):
    gesi_df = pd.read_csv('./data/csv/gesi.csv')
    gesi_df['date'] = pd.to_datetime(gesi_df['date'])
else:
    gesi_df = pd.DataFrame(columns=['date', 'gesi_value'])


# ==========================================
# 4. ОБЪЕДИНЕНИЕ ТАБЛИЦ (LEFT JOIN + FFILL)
# ==========================================

df = df.merge(block_df[['date', 'bsize']], on='date', how='left')
df = df.merge(hash_df[['date', 'hash-rate']], on='date', how='left')
df = df.merge(active_count_df[['date', 'active-count']], on='date', how='left')
df = df.merge(volume_sum_df[['date', 'total_fee']], on='date', how='left')
df = df.merge(transfers_df[['date', 'transfer_count']], on='date', how='left')

if not gesi_df.empty:
    df = df.merge(gesi_df[['date', 'gesi_value']], on='date', how='left')
else:
    df['gesi_value'] = np.nan

# Заполняем пропуски значениями из предыдущих дней
df = df.sort_values('date').ffill()
df = df.bfill()

# Сглаживание активных адресов за 14 дней
df['active-count_smoothed'] = df['active-count'].rolling(window=14).mean()

# Удаляем первые строки с NaN от скользящего среднего
df = df.dropna(subset=['active-count_smoothed']).reset_index(drop=True)


# ==========================================
# 5. ФОРМИРОВАНИЕ ИТОГОВОГО ДАТАСЕТА
# ==========================================

target_columns = [
    'date', 'close', 'volume', 'rsi', 'MACD_Cross_Power_Normalized', 
    'hash-rate', 'active-count', 'total_fee', 'transfer_count', 
    'active-count_smoothed', 'gesi_value', 'MACD', 'Signal_Line'
]

final_cols = [col for col in target_columns if col in df.columns]
data = df[final_cols].copy()

# Построение тепловой карты корреляции
plt.figure(figsize=(18, 16))
sns.heatmap(data.drop(columns=['date']).corr(), annot=True, cmap='Greens', fmt='.2f')
plt.savefig('no_corr_try.png')
plt.close()

# Сохранение и вывод
data.to_csv(main_data_path, index=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.info())
print(f"\n агрегация завершена. Итоговый датасет обновлен и сохранен в {main_data_path}")