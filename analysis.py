import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import fetch_coin_data
from datetime import datetime, timedelta
import os
import time
import logging
import requests
from config import (
    ETHERSCAN_API_KEY, BSCSCAN_API_KEY, POLYGONSCAN_API_KEY,
    SNOWSCAN_API_KEY, ARBISCAN_API_KEY, OPTIMISMSCAN_API_KEY,
    DEEPSEEK_API_KEY, DEXSCREENER_API, DEFILLAMA_API
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class EfficientLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Берем последний элемент последовательности
        out = self.linear(out)
        return out


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def get_ai_analysis(prompt):
    """Упрощенная версия без DeepSeek"""
    return "Анализ сгенерирован локально. Для расширенной аналитики настройте DeepSeek API."


def perform_full_analysis(coin):
    try:
        start_time = time.time()
        logger.info(f"🔍 Начало анализа для {coin}")

        # Получение данных
        logger.info("📥 Получение данных с CoinGecko...")
        df = fetch_coin_data(coin)

        if df.empty:
            raise ValueError("❌ Нет данных от CoinGecko API")

        if len(df) < 100:
            raise ValueError(f"❌ Недостаточно данных: только {len(df)} записей")

        logger.info(f"📊 Получено {len(df)} записей")

        # Нормализация данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['close']])

        # Создание последовательностей
        seq_length = 30  # Уменьшено для ускорения
        X, y = create_sequences(scaled_data, seq_length)

        # Разделение данных
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Проверка GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"⚙️ Используемое устройство: {device}")

        # Конвертация в тензоры
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Инициализация модели
        model = EfficientLSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # Увеличен learning rate

        # Обучение модели
        epochs = 50  # Уменьшено количество эпох
        batch_size = 64
        logger.info(f"🎓 Начало обучения: {epochs} эпох, размер батча {batch_size}")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"✅ Эпоха {epoch + 1}/{epochs} | Loss: {epoch_loss:.6f}")

        # Прогнозирование
        model.eval()
        with torch.no_grad():
            last_sequence = scaled_data[-seq_length:]
            last_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            next_day_pred = model(last_tensor).cpu().numpy()
            next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

        # Построение графика
        plt.figure(figsize=(12, 8))
        plt.plot(df.index, df['close'], 'b-', label='Исторические данные')

        # Прогноз
        next_day_date = datetime.now() + timedelta(days=1)
        plt.plot(next_day_date, next_day_price, 'ro', markersize=8, label=f'Прогноз: ${next_day_price:.2f}')

        plt.title(f'Анализ {coin.upper()}\nПрогноз: ${next_day_price:.2f}', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Цена (USD)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Сохранение графика
        os.makedirs('charts', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = f'charts/{coin}_{timestamp}.png'
        plt.savefig(chart_path, dpi=120)
        plt.close()

        logger.info(f"⏱️ Анализ завершен за {time.time() - start_time:.1f} сек.")
        return chart_path

    except Exception as e:
        logger.exception(f"❌ Критическая ошибка в анализе")
        raise