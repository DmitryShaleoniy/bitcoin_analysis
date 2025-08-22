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

with open('spizhennoe_avg_size.json', 'r', encoding='utf-8') as file: #здесь данные за последний год - каждый день
    data = json.load(file)
    block_df_temp = pd.DataFrame(data)
#print(data)
print('HERE')
block = block_df_temp['data']
print(block[0][0][0])
block_df_temp = pd.DataFrame()
block_df_temp['date'] = [i[0] for i in block[0]]
block_df_temp['bsize'] = [i[1] for i in block[0]]

block_df_temp['date'] = pd.to_datetime(block_df_temp['date'], unit='s')
print(block_df_temp.head())
print(block_df_temp.tail())

#block_df['data'] = pd.to_datetime(block_df['data'], unit='ms')

block_df_temp= block_df_temp.reset_index(drop=True)

with open('hash_rate_mean_stolen.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    hash_df = pd.DataFrame(data)
#print(data)
print(hash_df['v'].tail())

hash_df['t'] = pd.to_datetime(hash_df['t'], unit='s')

print(hash_df['t'].head())


hash_df = hash_df.rename(columns={'t': 'date'})
hash_df = hash_df.rename(columns={'v': 'hash-rate'})

hash_df= hash_df.reset_index(drop=True)

print('dfrsgrtherht')
print(df_no_time['date'].tail())
df_no_time['date'] = pd.to_datetime(df_no_time['date'])
print(df_no_time['date'].tail())


df1 = df_no_time.merge(block_df_temp, on='date', how='inner').sort_values(by='date') #inner - оставляем только те, которые есть в обоих датафреймах
df = df1.merge(hash_df, on='date', how='inner').sort_values(by='date') #то есть у нас есть только за ПОСЛЕДНИЙ ГОД

print(df.info())
print('HERE')
print(df['date'].tail())

metrics.append('bsize')
metrics.append('hash-rate')

metrics = [
    'close',
    'volume',
    'rsi',
    'EMA_12',
    'EMA_26',
    'MACD',
    'Signal_Line',
    'MACD_Cross_Power_Normalized',
    'hash-rate',
    'bsize'
]

plt.figure(figsize=(18, 16))
sns.heatmap( df[metrics].corr(),
             annot=True,
             cmap='coolwarm')
plt.savefig('correlation_heatmap_new.png')
plt.close()



# print('adefe')
#
print(df['date'].tail())
