import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from tinkoff.invest.utils import quotation_to_decimal

load_dotenv()
TOKEN = os.getenv("TINKOFF_TOKEN")

def fetch_h4_candles(figi: str, days_back: int = 30, use_sandbox: bool = False):
    """Загружает H4 свечи. При ошибке уменьшает период."""
    target = None if not use_sandbox else INVEST_GRPC_API_SANDBOX
    try:
        with Client(TOKEN, target=target) as client:
            from_ = datetime.now() - timedelta(days=days_back)
            to = datetime.now()
            print(f"Запрос свечей с {from_} по {to}")
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
            if not df.empty:
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
            return df
    except Exception as e:
        print(f"Ошибка при days_back={days_back}: {e}")
        if days_back > 10:
            print(f"Повторяем с меньшим периодом: {days_back-10} дней")
            return fetch_h4_candles(figi, days_back-10, use_sandbox)
        else:
            print("Не удалось загрузить данные даже за 10 дней.")
            return pd.DataFrame()

if __name__ == "__main__":
    # Правильный FIGI для акций Сбербанка (обыкновенные)
    figi = "BBG004730N88"
    print(f"Загрузка H4 свечей для FIGI: {figi}")
    df = fetch_h4_candles(figi, days_back=365, use_sandbox=False)

    if df.empty:
        print("❌ Данные не загружены.")
    else:
        print(f"✅ Загружено {len(df)} свечей.")
        print("Первые 5 строк:")
        print(df.head())
        df.to_csv("sber_h4.csv")
        print("💾 Данные сохранены в sber_h4.csv")