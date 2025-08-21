from datetime import date
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('df_analyss.csv')
metrics= ['close', 'volume', 'marketCap']


# print(df.info())
# print(df.columns)
# df_no_time = df.drop(columns=['timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name'])
# #преобразование даты
# df_no_time['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
# df_no_time['date'] = pd.to_datetime(df['timestamp']).dt.date
# df_no_time = df_no_time.drop(columns=['timestamp'])
#
# df_no_time = df_no_time[df_no_time['date'] >= date(2025, 1, 1)]
# df_no_time = df_no_time.reset_index(drop=True)

df_no_time = df
#рассчет среднего, прибыли и убытка
df_no_time['change'] = df_no_time['close'] - df_no_time['open']
df_no_time['gain']=df_no_time['change'].apply(lambda x: x  if x > 0 else 0)
df_no_time['loss']=df_no_time['change'].apply(lambda x: -x if x < 0 else 0)
#metrics.append('gain')
#metrics.append('loss')

df_no_time['gain_avg_14'] = df_no_time['gain'].rolling(14).mean()
df_no_time['loss_avg_14'] = df_no_time['loss'].rolling(14).mean()
metrics.append('gain_avg_14')
metrics.append('loss_avg_14')

#вводим rsi
df_no_time['rs'] = (df_no_time['gain_avg_14'] / df_no_time['loss_avg_14']).apply(lambda x: round(x, 2))
df_no_time['rsi'] = (100 - (100 / (1 + df_no_time['rs']))).apply(lambda x: round(x, 2))
metrics.append('rsi')
#rsi - это своеобразный спижометр для цены - rs считаем как отношение суммы ПРИРОСТОВ за 14 дней к сумме ПАДЕНИЙ за 14 дней
#--можно не за 14 дней--
#rsi - это приведение rs к процентному виду
#если rsi > 70 - актив перекуплен - это сигнал к падению
#если rsi < 30 - актив перепродан - это сигнал к росту

#метрики по rsi
df_no_time['is_overbought'] = (df_no_time['rsi'] > 70) * 1
df_no_time['is_oversold'] = (df_no_time['rsi'] < 30) * 1
metrics.append('is_overbought')
metrics.append('is_oversold')
#distance_from_70 = df['rsi_14'] - 70

df_no_time['EMA_12'] = df_no_time['close'].ewm(span=12, adjust=False).mean()
df_no_time['EMA_26'] = df_no_time['close'].ewm(span=26, adjust=False).mean()

metrics.append('EMA_12')
metrics.append('EMA_26')

##про EMA - это средняя цена за промежуток, но с акцентом на последние данные -
#последние данные влияют на результат больше, чем те, что в начале промежутка

#про MACD - это метрика, которая показиывает разницу (буквально разность) между двумя трендами -
#долгосрчный и короткосрочный - эти тренды представлеют собой EMA за 21 и 12 дней соответственно
#также существует такое понятие, как сигрнальная линия - это среднее значение MACD за последнее время(в нашем случае - за 9 дней)

#MACD исследуют на дистанции - смотрят на то, как он изменяется
#если он растет - разность между кратком и долгосрочным трендом растет - это означает бычий рост
#наоборот - это означает медвежий рост
#далее - про взаимосвязь с сигнальной линией MACD пересекает сигнальную снизу вверх — сигнал к покупке - у Макара тут это
# называется золотой крест (бычье пересечение)
#MACD пересекает сигнальную сверху вниз — сигнал к продажеMACD пересекает сигнальную снизу вверх) — сигнал к покупке
# - мертвый крест (медвежье пересечение).

df_no_time['MACD'] = df_no_time['EMA_12'] - df_no_time['EMA_26']
df_no_time['Signal_Line'] = df_no_time['MACD'].ewm(span=9, adjust=False).mean()
df_no_time['MACD_Histogram'] = df_no_time['MACD'] - df_no_time['Signal_Line']
metrics.append('MACD')
metrics.append('MACD_Histogram')
metrics.append( 'Signal_Line')

# Бычье пересечение (золотой крест MACD) в момент i
df_no_time['MACD_Bullish_Cross'] = ((df_no_time['MACD'] > df_no_time['Signal_Line']) &
                                    (df_no_time['MACD'].shift(1) <= df_no_time['Signal_Line'].shift(1))).astype(int)

# Медвежье пересечение (мертвый крест MACD) в момент i
df_no_time['MACD_Bearish_Cross'] = ((df_no_time['MACD'] < df_no_time['Signal_Line']) &
                                    (df_no_time['MACD'].shift(1) >= df_no_time['Signal_Line'].shift(1))).astype(int)

df_no_time['MACD_Cross_Power'] = df_no_time['MACD_Histogram']
#нормализовано относительно цены:
df_no_time['MACD_Cross_Power_Normalized'] = df_no_time['MACD_Histogram'] / df_no_time['close']
metrics.append('MACD_Cross_Power_Normalized')
metrics.append('MACD_Bullish_Cross')
metrics.append('MACD_Bearish_Cross')

sns.heatmap( df_no_time[metrics].corr(),
             annot=True,
             cmap='coolwarm')
plt.savefig('correlation_heatmap.png')
plt.close()

#print(df_no_time[metrics].head(15))
#print(df_no_time[metrics].tail(10))

print(df_no_time.info())

import json
with open('avg-block-size.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data)

with open('avg-block-size-year.json', 'r', encoding='utf-8') as file: #здесь данные за последний год - каждый день
    data = json.load(file)
    block_df = pd.DataFrame(data['avg-block-size'])
#print(data)
print(data.keys())
print(block_df.columns)
block_df['x'].head()
block_df['y'].tail()

block_df['x'] = pd.to_datetime(block_df['x'], unit='ms')

block_df = block_df.rename(columns={'x': 'date'})
block_df = block_df.rename(columns={'y': 'avg-block-size'})

block_df= block_df.reset_index(drop=True)

print(block_df.tail())
print(df_no_time.columns)
print(df_no_time['date'].tail())

print(df_no_time.info())
print(block_df.info())

df_no_time['date'] = pd.to_datetime(df_no_time['date'], format='%Y-%m-%d')

with open('hash-rate-year.json', 'r', encoding='utf-8') as file: #здесь данные за последний год - каждый день
    data = json.load(file)
    hash_df = pd.DataFrame(data['hash-rate'])
#print(data)
print(data.keys())
print(block_df.columns)
hash_df['x'].head()
hash_df['y'].tail()

hash_df['x'] = pd.to_datetime(hash_df['x'], unit='ms')

hash_df = hash_df.rename(columns={'x': 'date'})
hash_df = hash_df.rename(columns={'y': 'hash-rate'})

hash_df= hash_df.reset_index(drop=True)

hash_df.tail()

df1 = df_no_time.merge(block_df, on='date', how='inner').sort_values(by='date') #inner - оставляем только те, которые есть в обоих датафреймах
df = df1.merge(hash_df, on='date', how='inner').sort_values(by='date') #то есть у нас есть только за ПОСЛЕДНИЙ ГОД

df.info()
print(df['date'].tail())

metrics.append('avg-block-size')
metrics.append('hash-rate')

metrics = [
    'close',
    'volume',
    'marketCap',
    'rsi',
    'EMA_12',
    'EMA_26',
    'MACD',
    'Signal_Line',
    'MACD_Cross_Power_Normalized',
    'hash-rate',
    'avg-block-size'
]

plt.figure(figsize=(18, 16))
sns.heatmap( df[metrics].corr(),
             annot=True,
             cmap='coolwarm')
plt.savefig('correlation_heatmap_new.png')
plt.close()