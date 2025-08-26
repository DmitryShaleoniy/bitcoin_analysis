import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime as dt

import pmdarima as pm
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('features.csv')
df = df[['close', 'date']]
df['date'] = pd.to_datetime(df['date'])

# Фильтруем данные
df = df[df['date'] > dt.datetime(2024, 1, 1)]

df['derivative'] = pow(df['close'] - df['close'].shift(1), 4)
df = df.dropna()


result = adfuller(df['derivative'])
print('ADF Statistic: %lf' % result[0])
print('p-value: ',  result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

vav = 20

train, test = train_test_split(df['derivative'],
                               test_size=vav,
                               )

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Постройте графики ACF и PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train, lags=12, ax=ax1)
plot_pacf(train, lags=12, ax=ax2, method='ywm')
plt.savefig('acf_pacf.png')
plt.close()
#
#
# model = pm.auto_arima(
#     df['derivative'],  # ваш стационарный ряд
#     start_p=0,  # начальное значение p
#     start_q=0,  # начальное значение q
#     max_p=5,    # максимальное значение p
#     max_q=5,    # максимальное значение q
#     d=0,        # ряд уже стационарен, поэтому d=0
#     seasonal=False,  # нет сезонности
#     stepwise=True,   # пошаговый подбор
#     trace=True,      # вывод процесса подбора
#     error_action='ignore',
#     suppress_warnings=True,
#     information_criterion='aic'  # критерий для выбора модели
# )
#
# print(f"Лучшие параметры: p={model.order[0]}, d={model.order[1]}, q={model.order[2]}")
#
# import warnings
# from statsmodels.tsa.arima.model import ARIMA
# from tqdm import tqdm
#
# warnings.filterwarnings("ignore")  # Игнорируем предупреждения
#
# # Определите диапазоны для перебора
# p_range = range(0, 5)  # AR порядок
# q_range = range(0, 5)  # MA порядок
#
# best_aic = float('inf')
# best_order = None
# results = []
#
# # Перебор всех комбинаций
# for p, q in tqdm(itertools.product(p_range, q_range)):
#     try:
#         model = ARIMA(train, order=(p, 0, q))  # d=0 так как ряд стационарен
#         model_fit = model.fit()
#         aic = model_fit.aic
#
#         results.append({'p': p, 'q': q, 'aic': aic})
#
#         if aic < best_aic:
#             best_aic = aic
#             best_order = (p, q)
#
#     except Exception as e:
#         continue
#
# print(f"Лучший порядок: p={best_order[0]}, q={best_order[1]} с AIC={best_aic}")

y= df['derivative']
model = pm.auto_arima(train)
forecasts = model.predict(test.shape[0])
print(forecasts)

plt.figure(figsize=(10, 6))
# # plt.plot(df['date'], df['derivative'], color='blue', label='Derivative')
x = np.arange(y.shape[0])

plt.plot(x[:len(df)-vav], train, c='blue')
plt.plot(x[len(df)-vav:], forecasts, c='green')
plt.show()
plt.legend()
plt.xticks(rotation=90)
# plt.grid()

plt.savefig('target_regression.png')
#plt.close()