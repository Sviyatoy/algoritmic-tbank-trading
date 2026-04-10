import os
import pandas as pd
import uuid
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import lightgbm as lgb
from tinkoff.invest import Client, CandleInterval, OrderDirection, OrderType, MoneyValue
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from tinkoff.invest.utils import quotation_to_decimal

from features import add_technical_indicators, add_support_resistance, add_targets

load_dotenv()
TOKEN = os.getenv("TINKOFF_TOKEN")

FIGI = "BBG004730N88"
MODEL_PATH = "sber_lgb_model.txt"
THRESHOLD_PATH = "threshold.txt"
HORIZON = 4


def load_model_and_threshold():
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(THRESHOLD_PATH, 'r') as f:
        threshold = float(f.read().strip())
    return model, threshold


def get_last_candles(figi, count=60):
    with Client(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
        from_ = datetime.now() - timedelta(hours=count * 4 + 4)
        to = datetime.now()
        response = client.market_data.get_candles(
            figi=figi,
            from_=from_,
            to=to,
            interval=CandleInterval.CANDLE_INTERVAL_4_HOUR
        )
        candles = []
        for candle in response.candles:
            candles.append({
                'time': candle.time,
                'open': float(quotation_to_decimal(candle.open)),
                'high': float(quotation_to_decimal(candle.high)),
                'low': float(quotation_to_decimal(candle.low)),
                'close': float(quotation_to_decimal(candle.close)),
                'volume': candle.volume
            })
        df = pd.DataFrame(candles)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        return df


def prepare_features(df):
    df = add_technical_indicators(df)
    df = add_support_resistance(df)
    df = add_targets(df, horizons=[HORIZON])
    df = df.iloc[28:].copy()
    return df


def get_latest_features(df, feature_cols):
    latest = df[feature_cols].iloc[-1:].ffill().fillna(0)
    return latest.values


def get_feature_columns():
    model, _ = load_model_and_threshold()
    return model.feature_name()


def ensure_sandbox_account():
    """Открывает или возвращает существующий счёт в песочнице."""
    with Client(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
        # Используем users.get_accounts (вместо get_sandbox_accounts)
        accounts = client.users.get_accounts()
        if accounts.accounts:
            # У каждого счёта есть поле id (не account_id)
            account_id = accounts.accounts[0].id
        else:
            acc = client.sandbox.open_sandbox_account()
            account_id = acc.account_id
        return account_id


def fund_sandbox_account(account_id, amount_rub=100000):
    with Client(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
        client.sandbox.sandbox_pay_in(
            account_id=account_id,
            amount=MoneyValue(currency="rub", units=amount_rub, nano=0)
        )
        print(f"💰 Счёт пополнен на {amount_rub} RUB")




def place_sandbox_order(figi, quantity, account_id, direction='buy'):
    with Client(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
        order_dir = OrderDirection.ORDER_DIRECTION_BUY if direction == 'buy' else OrderDirection.ORDER_DIRECTION_SELL
        order_id = str(uuid.uuid4())  # правильный UUID4
        order = client.sandbox.post_sandbox_order(
            figi=figi,
            quantity=quantity,
            direction=order_dir,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=order_id
        )
        return order


def main():
    print(f"{datetime.now()} — Запуск торгового модуля")

    model, threshold = load_model_and_threshold()
    feature_cols = get_feature_columns()
    print(f"Модель загружена, порог={threshold:.3f}")

    df_candles = get_last_candles(FIGI, count=60)
    if len(df_candles) < 50:
        print("Недостаточно свечей для расчёта признаков")
        return

    df_features = prepare_features(df_candles)
    if df_features.empty:
        print("Ошибка при расчёте признаков")
        return

    X_latest = get_latest_features(df_features, feature_cols)
    proba = model.predict(X_latest)[0]
    signal = "buy" if proba > threshold else "hold"

    print(f"Вероятность роста: {proba:.3f}, сигнал: {signal.upper()}")

    if signal == "buy":
        account_id = ensure_sandbox_account()
        # Пополним счёт, если нужно (раскомментировать)
        fund_sandbox_account(account_id, 50000)

        last_price = df_features['close'].iloc[-1]
        quantity = int(10000 / last_price)
        if quantity > 0:
            try:
                order = place_sandbox_order(FIGI, quantity, account_id, "buy")
                print(f"✅ Заявка исполнена: {order}")
            except Exception as e:
                print(f"❌ Ошибка заявки: {e}")
        else:
            print("Недостаточно средств для покупки")
    else:
        print("Сигнал HOLD — заявка не выставлена")


if __name__ == "__main__":
    main()