import pandas as pd
import numpy as np


def add_technical_indicators(df):
    """Добавляет технические индикаторы с окнами до 28."""
    for period in [7, 14, 28]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'dist_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr_14'] / df['close'] * 100

    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_sma'] = obv.rolling(window=14).mean()

    # Объём
    df['volume_sma'] = df['volume'].rolling(window=14).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Свечные паттерны
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    df['body_ratio'] = body / (df['high'] - df['low'] + 1e-6)
    df['doji'] = (body < (df['high'] - df['low']) * 0.1).astype(int)
    df['hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body)).astype(int)

    return df


def add_support_resistance(df):
    """Упрощённые уровни: расстояние до ближайшего локального экстремума."""
    peaks = df['high'][(df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))]
    troughs = df['low'][(df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))]

    if len(peaks) > 0:
        peak_list = peaks.sort_values().tolist()
        df['dist_to_resistance'] = df['close'].apply(lambda x: min([p - x for p in peak_list if p > x], default=np.nan))
        df['resistance_ratio'] = df['dist_to_resistance'] / df['close'] * 100
    else:
        df['dist_to_resistance'] = np.nan
        df['resistance_ratio'] = np.nan

    if len(troughs) > 0:
        trough_list = troughs.sort_values().tolist()
        df['dist_to_support'] = df['close'].apply(lambda x: min([x - t for t in trough_list if t < x], default=np.nan))
        df['support_ratio'] = df['dist_to_support'] / df['close'] * 100
    else:
        df['dist_to_support'] = np.nan
        df['support_ratio'] = np.nan

    return df


def add_targets(df, horizons=[1, 2, 4]):
    for h in horizons:
        df[f'future_return_{h}'] = (df['close'].shift(-h) / df['close'] - 1) * 100
        df[f'target_up_{h}'] = (df[f'future_return_{h}'] > 0.3).astype(int)
        df[f'target_down_{h}'] = (df[f'future_return_{h}'] < -0.3).astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("sber_h4.csv", index_col="time", parse_dates=True)
    print(f"Исходные данные: {df.shape[0]} строк, {df.shape[1]} колонок")

    df = add_technical_indicators(df)
    print("Добавлены индикаторы")

    df = add_support_resistance(df)
    print("Добавлены S/R")

    df = add_targets(df, horizons=[1, 2, 4])
    print("Добавлены цели")

    # Удаляем только последние 4 строки (максимальный горизонт) и строки с NaN в целях
    max_h = 4
    df_clean = df.iloc[:-max_h].copy()
    df_clean = df_clean.dropna(subset=[f'target_up_{h}' for h in [1, 2, 4]])

    print(f"Очищенные данные: {df_clean.shape[0]} строк")

    if df_clean.shape[0] > 50:
        df_clean.to_csv("sber_h4_features.csv")
        print("💾 Сохранено в sber_h4_features.csv")
        for h in [1, 2, 4]:
            up = df_clean[f'target_up_{h}'].mean()
            down = df_clean[f'target_down_{h}'].mean()
            print(f"Горизонт {h} свечей: Up={up:.2%}, Down={down:.2%}")
    else:
        print("❌ Недостаточно данных. Нужно минимум 50 строк для обучения.")
        print("Попробуйте загрузить больше исторических свечей (180+ дней).")