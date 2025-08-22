from datetime import date
from threading import activeCount

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('df_analyss.csv')

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


df_no_time['gain_avg_14'] = df_no_time['gain'].rolling(14).mean()
df_no_time['loss_avg_14'] = df_no_time['loss'].rolling(14).mean()


#вводим rsi
df_no_time['rs'] = (df_no_time['gain_avg_14'] / df_no_time['loss_avg_14']).apply(lambda x: round(x, 2))
df_no_time['rsi'] = (100 - (100 / (1 + df_no_time['rs']))).apply(lambda x: round(x, 2))

#rsi - это своеобразный спижометр для цены - rs считаем как отношение суммы ПРИРОСТОВ за 14 дней к сумме ПАДЕНИЙ за 14 дней
#--можно не за 14 дней--
#rsi - это приведение rs к процентному виду
#если rsi > 70 - актив перекуплен - это сигнал к падению
#если rsi < 30 - актив перепродан - это сигнал к росту

#метрики по rsi
df_no_time['is_overbought'] = (df_no_time['rsi'] > 70) * 1
df_no_time['is_oversold'] = (df_no_time['rsi'] < 30) * 1
#distance_from_70 = df['rsi_14'] - 70

df_no_time['EMA_12'] = df_no_time['close'].ewm(span=12, adjust=False).mean()
df_no_time['EMA_26'] = df_no_time['close'].ewm(span=26, adjust=False).mean()

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

# Бычье пересечение (золотой крест MACD) в момент i
df_no_time['MACD_Bullish_Cross'] = ((df_no_time['MACD'] > df_no_time['Signal_Line']) &
                                    (df_no_time['MACD'].shift(1) <= df_no_time['Signal_Line'].shift(1))).astype(int)

# Медвежье пересечение (мертвый крест MACD) в момент i
df_no_time['MACD_Bearish_Cross'] = ((df_no_time['MACD'] < df_no_time['Signal_Line']) &
                                    (df_no_time['MACD'].shift(1) >= df_no_time['Signal_Line'].shift(1))).astype(int)

df_no_time['MACD_Cross_Power'] = df_no_time['MACD_Histogram']
#нормализовано относительно цены:
df_no_time['MACD_Cross_Power_Normalized'] = df_no_time['MACD_Histogram'] / df_no_time['close']


import json

with open('spizhennoe_avg_size.json', 'r', encoding='utf-8') as file: #здесь данные за последний год - каждый день
    data = json.load(file)
    block_df_temp = pd.DataFrame(data)

block = block_df_temp['data']
block_df_temp = pd.DataFrame()

block_df_temp['date'] = [i[0] for i in block[0]]
block_df_temp['bsize'] = [i[1] for i in block[0]]
block_df_temp['date'] = pd.to_datetime(block_df_temp['date'], unit='s')
block_df_temp= block_df_temp.reset_index(drop=True)


with open('hash_rate_mean_stolen.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    hash_df = pd.DataFrame(data)

hash_df['t'] = pd.to_datetime(hash_df['t'], unit='s')

hash_df = hash_df.rename(columns={'t': 'date'})
hash_df = hash_df.rename(columns={'v': 'hash-rate'})
hash_df= hash_df.reset_index(drop=True)


#активные адреса
#https://studio.glassnode.com/charts/addresses.ActiveCount?a=BTC&chartStyle=column&pScl=lin&zoom=all
with open('active_count.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    active_count_df = pd.DataFrame(data)

active_count_df['t'] = pd.to_datetime(active_count_df['t'], unit='s')
active_count_df = active_count_df.rename(columns={'t': 'date'})
active_count_df = active_count_df.rename(columns={'v': 'active-count'})
active_count_df= active_count_df.reset_index(drop=True)


#датасет с суммой всех fees за день (крутой, мало коррелирует)
#https://studio.glassnode.com/charts/fees.VolumeSum?a=BTC&chartStyle=column&pScl=lin&zoom=all
with open('volume_sum.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    total_fee_df = pd.DataFrame(data)

total_fee_df['t'] = pd.to_datetime(total_fee_df['t'], unit='s')

total_fee_df = total_fee_df.rename(columns={'t': 'date'})
total_fee_df = total_fee_df.rename(columns={'v': 'total_fee'})
total_fee_df= total_fee_df.reset_index(drop=True)


#монет через транзакции (мб иожно улучщить за счет сравнения с ценой монеты???)
#https://api.glassnode.com/v1/metrics/transactions/transfers_volume_sum?a=BTC&i=24h
with open('transfers_volume_sum.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    transfer_count_df = pd.DataFrame(data)

transfer_count_df['t'] = pd.to_datetime(transfer_count_df['t'], unit='s')

transfer_count_df = transfer_count_df.rename(columns={'t': 'date'})
transfer_count_df = transfer_count_df.rename(columns={'v': 'transfer_count'})
transfer_count_df= transfer_count_df.reset_index(drop=True)


df_no_time['date'] = pd.to_datetime(df_no_time['date'])
df1 = df_no_time.merge(block_df_temp, on='date', how='inner').sort_values(by='date') #inner - оставляем только те, которые есть в обоих датафреймах
df = df1.merge(hash_df, on='date', how='inner').sort_values(by='date') #то есть у нас есть только за ПОСЛЕДНИЙ ГОД
df = df.merge(active_count_df, on='date', how='inner').sort_values(by='date')
df = df.merge(total_fee_df, on='date', how='inner').sort_values(by='date')
df = df.merge(transfer_count_df, on='date', how='inner').sort_values(by='date')

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
    'bsize',
    'active-count',
    'total_fee',
    'transfer_count'
]

plt.figure(figsize=(18, 16))
sns.heatmap( df[metrics].corr(),
             annot=True,
             cmap='coolwarm')
plt.savefig('correlation_heatmap_new.png')
plt.close()



# print('adefe')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.info())
print(df.head())
print(df.tail())


data = df[[ 'date','close', 'volume', 'rsi', 'MACD', 'hash-rate', 'bsize', 'active-count', 'total_fee', 'transfer_count']]
plt.figure(figsize=(18, 16))
sns.heatmap( data.drop(columns=['date']).corr(),
             annot=True,
             cmap='Greens')
plt.savefig('no_corr_try.png')
plt.close()
