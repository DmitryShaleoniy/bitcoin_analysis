import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from scipy. stats import boxcox


df = pd.read_csv('features.csv')
# df['target'] = df['close'].shift(-1).values
# df['transformed_target'] = np.log(df['target'])
df = df[['close', 'date']]
df['derivative'] = df['close'] - df['close'].shift(1)
print(df)
df = df.dropna()
# df['transformed_target'] = np.log(df['derivative'])
# y = df[['target']].values
# print(df[['date', 'close', 'target']].tail())

tt = TransformedTargetRegressor(
    regressor=Ridge(),
    func=np.log,
    inverse_func=np.exp
)
result = adfuller(df['derivative'])
print('ADF Statistic: %lf' % result[0])
print('p-value: ',  result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

plt.figure(figsize=(10, 6))
# plt.plot(df['date'], df['target'], color='green', label='Target')
# plt.plot(df['date'], df['transformed_target'], color='red', label='Transformed Target')
plt.plot(df['date'], df['derivative'], color='blue', label='Derivative')
# plt.plot(result)
# plt.plot(df['date'], df['transformed_target'], color='red', label='Transformed Derivative')

plt.legend()
plt.xticks(rotation=90)
# plt.grid()

plt.savefig('target_regression.png')
plt.close()