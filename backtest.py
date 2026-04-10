import pandas as pd
import numpy as np
import lightgbm as lgb

# Загрузка данных
df = pd.read_csv("sber_h4_features.csv", index_col=0, parse_dates=True)
print(f"Загружено {len(df)} строк")

# Модель и порог
model = lgb.Booster(model_file='sber_lgb_model.txt')
with open('threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

# Признаки
feature_cols = model.feature_name()
print(f"Используем {len(feature_cols)} признаков")

# Подготовка X и y
X = df[feature_cols].ffill().fillna(0)
y_true = df['target_up_4'].fillna(0).astype(int)

# Прогноз вероятностей
y_proba = model.predict(X)

# Сигналы (1 если вероятность > порога)
signals = (y_proba > threshold).astype(int)

# Ограничимся строками, где y_true не NaN (т.е. есть будущие данные)
valid_mask = ~df['target_up_4'].isna()
y_true_valid = y_true[valid_mask]
signals_valid = signals[valid_mask]
y_proba_valid = y_proba[valid_mask]

# Метрики
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_true_valid, signals_valid)
prec = precision_score(y_true_valid, signals_valid, zero_division=0)
rec = recall_score(y_true_valid, signals_valid, zero_division=0)
f1 = f1_score(y_true_valid, signals_valid, zero_division=0)

print(f"\n=== Качество модели на исторических данных ===")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1: {f1:.3f}")
print(f"Доля сигналов BUY: {signals_valid.mean():.2%}")

# ===== Симуляция торговли =====
initial_capital = 100_000.0
capital = initial_capital
position = 0
trade_log = []
close_prices = df['close'].values
dates = df.index.values

# Держим позицию фиксированное количество свечей (например, 4)
HOLD_HORIZON = 4
entry_idx = None

for i in range(len(df)):
    if i >= len(signals):
        break
    # Если есть открытая позиция, проверяем, не пора ли закрыть
    if position > 0 and entry_idx is not None and (i - entry_idx) >= HOLD_HORIZON:
        # Закрываем по текущей цене
        sell_price = close_prices[i]
        capital += position * sell_price
        trade_log.append({
            'date': dates[i],
            'action': 'SELL',
            'price': sell_price,
            'shares': position,
            'capital': capital,
            'position': 0,
            'return_pct': (sell_price / entry_price - 1) * 100 if entry_price else 0
        })
        position = 0
        entry_idx = None
        entry_price = None

    # Сигнал на покупку (только если нет открытой позиции)
    if position == 0 and i < len(signals) and signals[i] == 1:
        price = close_prices[i]
        if capital > 0:
            shares = int(capital / price)
            if shares > 0:
                cost = shares * price
                capital -= cost
                position = shares
                entry_idx = i
                entry_price = price
                trade_log.append({
                    'date': dates[i],
                    'action': 'BUY',
                    'price': price,
                    'shares': shares,
                    'capital': capital,
                    'position': position,
                    'return_pct': 0
                })

# Если осталась открытая позиция в конце, закрываем по последней цене
if position > 0:
    last_price = close_prices[-1]
    capital += position * last_price
    trade_log.append({
        'date': dates[-1],
        'action': 'SELL',
        'price': last_price,
        'shares': position,
        'capital': capital,
        'position': 0,
        'return_pct': (last_price / entry_price - 1) * 100 if entry_price else 0
    })

final_capital = capital
total_return = (final_capital / initial_capital - 1) * 100

print(f"\n=== Результаты бэктеста (1 акция, без плеча, удержание {HOLD_HORIZON} свечей) ===")
print(f"Начальный капитал: {initial_capital:,.0f} RUB")
print(f"Конечный капитал: {final_capital:,.0f} RUB")
print(f"Доходность: {total_return:.2f}%")
print(f"Количество сделок (покупок): {len([t for t in trade_log if t['action']=='BUY'])}")

# Сохраним лог
if trade_log:
    pd.DataFrame(trade_log).to_csv("backtest_trades.csv", index=False)
    print("💾 Лог сделок сохранён в backtest_trades.csv")