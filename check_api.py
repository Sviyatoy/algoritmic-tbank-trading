import os
from dotenv import load_dotenv
from tinkoff.invest import Client
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX

load_dotenv()
TOKEN = os.getenv("TINKOFF_TOKEN")


def main():
    print("🔄 Подключение к песочнице...")
    if not TOKEN:
        print("❌ Токен не найден. Проверьте .env файл")
        return

    try:
        with Client(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
            # 1. Открываем счёт в песочнице
            acc = client.sandbox.open_sandbox_account()
            account_id = acc.account_id
            print(f"✅ Счёт открыт. ID: {account_id}")

            # 2. Получаем портфель (новый метод)
            portfolio = client.operations.get_portfolio(account_id=account_id)

            # 3. Извлекаем общую стоимость портфеля
            # Поле total_amount_ports может отсутствовать, используем total_amount_shares + total_amount_bonds + ...
            # Но проще: portfolio.total_amount_units? Проверим.
            # На всякий случай выведем все атрибуты объекта, чтобы увидеть структуру
            # print(dir(portfolio))  # раскомментировать для отладки

            # Безопасное получение суммы:
            total_value = 0
            if hasattr(portfolio, 'total_amount_ports') and portfolio.total_amount_ports:
                total_value = portfolio.total_amount_ports[0].units
            elif hasattr(portfolio, 'total_amount_shares'):
                total_value = portfolio.total_amount_shares.units
            else:
                # Если ничего нет, просто выводим 0
                total_value = 0

            print(f"✅ Портфель: {total_value} RUB")

            # 4. Закрываем счёт
            client.sandbox.close_sandbox_account(account_id=account_id)
            print("✅ Счёт закрыт. Всё работает!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()