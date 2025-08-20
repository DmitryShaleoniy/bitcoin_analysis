from datetime import date

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('BTC_merged_2010_to_2025.csv')a
metrics= ['close', 'volume', 'marketCap']


print(df.info())
print(df.columns)
df_no_time = df.drop(columns=['timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name'])
#преобразование даты
df_no_time['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_no_time['date'] = pd.to_datetime(df['timestamp']).dt.date
df_no_time = df_no_time.drop(columns=['timestamp'])

df_no_time = df_no_time[df_no_time['date'] >= date(2025, 1, 1)]
df_no_time = df_no_time.reset_index(drop=True)

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
#метрики по rsi
df_no_time['is_overbought'] = (df_no_time['rsi'] > 70) * 1
df_no_time['is_oversold'] = (df_no_time['rsi'] < 30) * 1
metrics.append('is_overbought')
metrics.append('is_oversold')
#distance_from_70 = df['rsi_14'] - 70

df_no_time['EMA_12'] = df_no_time['close'].ewm(span=12, adjust=False).mean()
df_no_time['EMA_26'] = df_no_time['close'].ewm(span=26, adjust=False).mean()
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
