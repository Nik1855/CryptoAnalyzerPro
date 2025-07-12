import logging
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes
from analysis import perform_full_analysis
from config import TELEGRAM_BOT_TOKEN
import os
import asyncio

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(
        f"🚀 Привет, {user.mention_html()}!\n\n"
        "Я - CryptoAnalyzerPro, ваш персональный аналитик криптовалют.\n"
        "Используйте команды:\n"
        "/analyze bitcoin - анализ криптовалюты\n"
        "/help - помощь по использованию"
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("❌ Укажите криптовалюту: /analyze bitcoin")
            return

        coin = context.args[0].lower()
        msg = await update.message.reply_text(f"🔍 Начинаю анализ {coin.upper()}...")

        # Запускаем анализ в отдельном потоке
        loop = asyncio.get_running_loop()
        image_path = await loop.run_in_executor(None, perform_full_analysis, coin)

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=InputFile(photo),
                    caption=f"📊 Результат анализа {coin.upper()}"
                )
            os.remove(image_path)
            await msg.delete()  # Удалить промежуточное сообщение
        else:
            await update.message.reply_text("❌ Ошибка генерации графика")

    except Exception as e:
        logger.error(f"Ошибка в analyze_command: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 Список доступных команд:\n\n"
        "/start - начать работу с ботом\n"
        "/analyze [криптовалюта] - анализ криптовалюты\n"
        "/help - показать это сообщение\n\n"
        "Пример: /analyze bitcoin"
    )
    await update.message.reply_text(help_text)


def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("help", help_command))

    logger.info("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()