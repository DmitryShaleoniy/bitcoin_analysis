import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('main_data.csv')
# Убедимся, что датафрейм df содержит нужные колонки
data_for_plot = df[['date', 'close', 'active-count','active-count_smoothed']]

# Создаем фигуру и ось
plt.figure(figsize=(14, 6))

# График цены закрытия (close)
plt.plot(data_for_plot['date'], data_for_plot['close'], label='Цена закрытия (close)', color='blue')

# График количества активных адресов (active-count)
plt.plot(data_for_plot['date'], data_for_plot['active-count'], label='Активные адреса (active-count)', color='green')
plt.plot(data_for_plot['date'], data_for_plot['active-count_smoothed'], label='Активные адреса (active-count_smoothed)', color='red')

# Настройки графика
plt.title('Сравнение цены закрытия и количества активных адресов')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

# Поворот меток дат для удобства чтения
plt.gcf().autofmt_xdate()

# Сохраняем график
plt.savefig('close_active_count.png')
plt.close()