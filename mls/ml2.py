import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Улучшенная подготовка данных с акцентом на стационарность
def prepare_data_advanced_stationary(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # Базовые признаки с акцентом на стационарность
    df['log_close'] = np.log(df['close'])
    df['log_return'] = df['log_close'].diff()

    # Процентные изменения вместо абсолютных значений
    for col in ['volume', 'hash-rate', 'active-count', 'total_fee', 'transfer_count']:
        df[f'{col}_pct_change'] = df[col].pct_change().fillna(0)

    # RSI с различными периодами
    for rsi_period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))

    # MACD с различными параметрами
    for fast, slow in [(12, 26), (8, 17), (5, 35)]:
        exp1 = df['close'].ewm(span=fast).mean()
        exp2 = df['close'].ewm(span=slow).mean()
        df[f'MACD_{fast}_{slow}'] = exp1 - exp2
        df[f'MACD_signal_{fast}_{slow}'] = df[f'MACD_{fast}_{slow}'].ewm(span=9).mean()
        df[f'MACD_hist_{fast}_{slow}'] = df[f'MACD_{fast}_{slow}'] - df[f'MACD_signal_{fast}_{slow}']

    # Волатильность различных периодов
    for window in [7, 14, 21, 30]:
        df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std().fillna(0)

    # Объемные индикаторы
    df['volume_ma_ratio_7_21'] = df['volume'].rolling(7).mean() / df['volume'].rolling(21).mean()
    df['volume_ma_ratio_14_50'] = df['volume'].rolling(14).mean() / df['volume'].rolling(50).mean()

    # Целевая переменная - направление через 3 дня (среднесрочный тренд)
    df['future_log_return_3'] = df['log_return'].shift(-3)
    df['target'] = (df['future_log_return_3'] > 0).astype(int)

    # Удаляем строки с NaN
    df = df.dropna()

    return df

# Создание лаговых признаков с отбором
def create_selected_lags(df, selected_features, n_lags=5):
    for feature in selected_features:
        for lag in range(1, n_lags+1):
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    return df.dropna()

# Отбор наиболее важных признаков
def select_features(X, y, n_features=30):
    from sklearn.ensemble import RandomForestClassifier

    # Быстрый отбор признаков с помощью RandomForest
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Получаем важность признаков
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Выбираем топ-N признаков
    selected_features = importance.head(n_features)['feature'].values

    return selected_features

# Построение улучшенных моделей с балансировкой классов
def build_improved_models():
    # Базовые модели
    models = {
        'gbc': GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=150,
            subsample=0.8,
            random_state=42
        ),
        'rfc': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
    }

    # Создаем пайплайны с балансировкой классов
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = ImbPipeline([
            ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
            ('model', model)
        ])

    return pipelines

# Улучшенная оценка моделей
def evaluate_improved_models(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    all_predictions = {}

    models = build_improved_models()

    for name, pipeline in models.items():
        metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'roc_auc': [], 'profit_potential': []
        }
        predictions = []
        probabilities = []
        actuals = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                # Обучение пайплайна с SMOTE
                pipeline.fit(X_train_scaled, y_train)

                # Предсказания
                preds = pipeline.predict(X_test_scaled)
                probas = pipeline.predict_proba(X_test_scaled)[:, 1]

                # Сохраняем для анализа
                predictions.extend(preds)
                probabilities.extend(probas)
                actuals.extend(y_test.values)

                # Метрики
                metrics['accuracy'].append(accuracy_score(y_test, preds))
                metrics['precision'].append(precision_score(y_test, preds, zero_division=0))
                metrics['recall'].append(recall_score(y_test, preds, zero_division=0))
                metrics['f1'].append(f1_score(y_test, preds, zero_division=0))
                metrics['roc_auc'].append(roc_auc_score(y_test, probas))

                # Потенциальная прибыльность
                price_changes = np.abs(X_test['log_return'].values)
                profit = np.where(preds == y_test, price_changes, -price_changes)
                metrics['profit_potential'].append(np.mean(profit))

            except Exception as e:
                print(f"Ошибка в {name}: {e}")
                continue

        if metrics['accuracy']:
            results[name] = {key: np.mean(values) for key, values in metrics.items()}
            all_predictions[name] = {
                'predictions': np.array(predictions),
                'probabilities': np.array(probabilities),
                'actuals': np.array(actuals)
            }

    return results, all_predictions

# Визуализация результатов с акцентом на торговую стратегию
def plot_trading_strategy_results(results, predictions_dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # График 1: Сравнение метрик
    ax = axes[0, 0]
    model_names = list(results.keys())
    metrics_to_plot = ['accuracy', 'recall', 'precision', 'f1']

    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        values = [results[name][metric] for name in model_names]
        ax.bar(x + i*width, values, width, label=metric)

    ax.set_xlabel('Модели')
    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение метрик моделей')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 2: ROC-AUC
    ax = axes[0, 1]
    roc_auc_scores = [results[name]['roc_auc'] for name in model_names]

    bars = ax.bar(model_names, roc_auc_scores)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Случайное угадывание')
    ax.set_title('ROC-AUC по моделям')
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0, 1)

    for bar, auc in zip(bars, roc_auc_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom')

    # График 3: Прибыльность
    ax = axes[1, 0]
    profit_scores = [results[name]['profit_potential'] for name in model_names]

    bars = ax.bar(model_names, profit_scores)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Потенциальная прибыльность')
    ax.set_ylabel('Средняя прибыль на сделку')

    for bar, profit in zip(bars, profit_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{profit:.6f}', ha='center', va='bottom')

    # График 4: Кумулятивная доходность стратегии
    ax = axes[1, 1]
    for name in model_names:
        if name in predictions_dict:
            preds = predictions_dict[name]['predictions']
            actuals = predictions_dict[name]['actuals']
            price_changes = np.abs(np.random.randn(len(actuals))) * 0.01  # Замените на реальные изменения цены

            # Симуляция торговой стратегии
            returns = np.where(preds == actuals, price_changes, -price_changes)
            cumulative_returns = np.cumsum(returns)

            ax.plot(cumulative_returns, label=name)

    ax.set_title('Кумулятивная доходность торговой стратегии')
    ax.set_xlabel('Время')
    ax.set_ylabel('Кумулятивная доходность')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Основной пайплайн
def main_improved():
    # Загрузка данных
    df = pd.read_csv('main_data.csv')

    # Улучшенная подготовка данных
    df = prepare_data_advanced_stationary(df)

    # Выбор признаков для лагов
    base_features = [
        'log_return', 'volume_pct_change', 'hash-rate_pct_change',
        'active-count_pct_change', 'total_fee_pct_change', 'transfer_count_pct_change',
        'RSI_7', 'RSI_14', 'RSI_21', 'MACD_12_26', 'MACD_hist_12_26',
        'volatility_7', 'volatility_14', 'volatility_21'
    ]

    # Создание лагов
    df_lagged = create_selected_lags(df, base_features, n_lags=5)

    # Целевая переменная и признаки
    target = df_lagged['target']
    features = df_lagged.drop(['target', 'future_log_return_3', 'log_close'], axis=1)

    # Отбор наиболее важных признаков
    selected_features = select_features(features, target, n_features=40)
    X = features[selected_features]
    y = target

    print(f"Размер данных: {X.shape}")
    print(f"Баланс классов: {y.mean():.3f} (доля растущих периодов)")
    print(f"Отобрано признаков: {len(selected_features)}")

    # Оценка моделей
    results, all_predictions = evaluate_improved_models(X, y)

    # Выбор лучшей модели
    best_model_name = max(results, key=lambda x: results[x]['profit_potential'])
    print(f"\nЛучшая модель: {best_model_name}")
    for metric, value in results[best_model_name].items():
        print(f"{metric}: {value:.4f}")

    # Визуализация
    plot_trading_strategy_results(results, all_predictions)

    # Финальное обучение лучшей модели
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    models = build_improved_models()
    best_model = models[best_model_name]
    best_model.fit(X_scaled, y)

    return best_model, scaler, selected_features, results, all_predictions

if __name__ == "__main__":
    model, scaler, feature_names, results, predictions = main_improved()