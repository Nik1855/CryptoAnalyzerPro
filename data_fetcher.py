import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import os
from config import DEXSCREENER_API, DEFILLAMA_API

logger = logging.getLogger(__name__)


def fetch_coin_data(coin):
    try:
        # Кэширование
        cache_dir = "data_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{coin}_data.csv"

        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < 86400:  # 24 часа
                return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')

        # Получение данных
        logger.info(f"🌐 Запрос данных для {coin}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 год данных

        url = "https://api.coingecko.com/api/v3/coins/market_chart/range"
        params = {
            'id': coin,
            'vs_currency': 'usd',
            'from': int(start_date.timestamp()),
            'to': int(end_date.timestamp())
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Обработка данных
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')
        df = df.resample('D').last().ffill()
        df.rename(columns={'price': 'close'}, inplace=True)
        df.drop(columns=['timestamp'], inplace=True)

        # Сохранение в кэш
        df.to_csv(cache_file)
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"⚠️ Ошибка сети: {e}")
    except Exception as e:
        logger.error(f"⚠️ Ошибка обработки данных: {e}")

    # Возврат кэшированных данных в случае ошибки
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
    return pd.DataFrame()