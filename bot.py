import os
import logging
import asyncio
from typing import List, Dict, Any
from telegram import Update, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_pipeline import RAGPipeline
from config import TELEGRAM_BOT_TOKEN, CHAT_SYSTEM_PROMPT, MAX_HISTORY_LENGTH, RAG_BACKEND

# Добавлена поддержка вебхуков для развертывания на сервере
import ssl
from aiohttp import web

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RAGTelegramBot:
    def __init__(self):
        if RAG_BACKEND == "llamaindex":
            from llamaindex_pipeline import LlamaIndexRAG
            self.rag_pipeline = LlamaIndexRAG()
        else:
            self.rag_pipeline = RAGPipeline()
        self.user_sessions = {}  # Для хранения сессий пользователей
        self.chat_histories = {}  # Для хранения истории чатов

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🤖 Привет! Я RAG-бот с локальными моделями.

Доступные команды:
/start - это сообщение
/add_docs - добавить документы в базу знаний
/ask [вопрос] - задать вопрос по документам
/stats - статистика базы знаний
/reset - очистить текущую сессию

Просто отправь мне документ (PDF, TXT, DOCX) или задай вопрос!
        """
        await update.message.reply_text(welcome_text)

    async def add_documents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /add_docs"""
        await update.message.reply_text(
            "📎 Отправь мне документы (PDF, TXT, DOCX) для добавления в базу знаний. "
            "Я обработаю их и сохраню в векторной базе."
        )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка входящих документов"""
        document: Document = update.message.document
        user_id = update.effective_user.id

        try:
            # Создаем папку для временных файлов пользователя
            user_temp_dir = f"temp/{user_id}"
            os.makedirs(user_temp_dir, exist_ok=True)

            # Скачиваем документ
            file = await document.get_file()
            file_path = f"{user_temp_dir}/{document.file_name}"
            await file.download_to_drive(file_path)

            # Проверяем, что файл действительно скачался
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                await update.message.reply_text("❌ Ошибка при скачивании файла")
                return

            await update.message.reply_text(
                f"📥 Документ '{document.file_name}' получен. Начинаю обработку..."
            )

            # Проверяем размер файла (ограничение Telegram - 20MB, но для обработки разумно ограничить)
            if document.file_size and document.file_size > 10 * 1024 * 1024:  # 10MB
                await update.message.reply_text("❌ Файл слишком большой (максимум 10MB)")
                return

            # Обрабатываем документ через RAG пайплайн
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.rag_pipeline.add_document, file_path
                )

                if "error" in result:
                    await update.message.reply_text(f"❌ Ошибка при обработке документа: {result['error']}")
                else:
                    await update.message.reply_text(
                        f"✅ Документ обработан!\n"
                        f"• Извлечено чанков: {result['chunks_count']}\n"
                        f"• Размер базы: {result['total_chunks']} чанков"
                    )
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                await update.message.reply_text("❌ Ошибка при обработке документа")

            # Очищаем временный файл
            try:
                os.remove(file_path)
            except:
                pass  # Игнорируем ошибку удаления файла

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            await update.message.reply_text("❌ Ошибка при обработке документа")

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        text = update.message.text

        try:
            # Отправляем промежуточное сообщение
            processing_msg = await update.message.reply_text("🤔 Обрабатываю ваш запрос...")

            # Проверяем, есть ли документы в базе знаний
            stats = self.rag_pipeline.get_collection_stats()
            if stats['total_chunks'] > 0:
                # Если есть документы, используем RAG
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.rag_pipeline.ask_question, text, update.effective_user.id
                )

                # Форматируем ответ
                answer_text = f"**Ответ:** {response['answer']}\n\n"

                if response['sources']:
                    answer_text += "**Источники:**\n"
                    for i, source in enumerate(response['sources'][:3], 1):
                        answer_text += f"{i}. {source['source']} (сходство: {source['score']:.2f})\n"

                # Редактируем сообщение с финальным ответом
                await processing_msg.edit_text(answer_text)
            else:
                # Если нет документов, используем обычный чат с Ollama
                import ollama

                user_id = update.effective_user.id

                # Инициализируем историю чата для пользователя, если не существует
                if user_id not in self.chat_histories:
                    self.chat_histories[user_id] = []

                # Добавляем сообщение пользователя в историю
                self.chat_histories[user_id].append({"role": "user", "content": text})

                # Формируем промпт с историей
                messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
                messages.extend(self.chat_histories[user_id][-MAX_HISTORY_LENGTH:])  # Ограничиваем историю

                try:
                    response = ollama.chat(
                        model=self.rag_pipeline.llm_model,
                        messages=messages,
                        options={
                            'temperature': 0.7,
                            'top_k': 40,
                            'top_p': 0.9,
                        }
                    )

                    ai_response = response['message']['content']

                    # Добавляем ответ AI в историю
                    self.chat_histories[user_id].append({"role": "assistant", "content": ai_response})

                    # Редактируем сообщение с финальным ответом
                    await processing_msg.edit_text(ai_response)
                except Exception as e:
                    logger.error(f"Error with Ollama chat: {e}")
                    await processing_msg.edit_text("Извините, сейчас не могу обработать ваш запрос.")

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            await update.message.reply_text("❌ Ошибка при обработке сообщения")

    async def ask_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /ask"""
        if not context.args:
            await update.message.reply_text("❌ Использование: /ask [ваш вопрос]")
            return

        question = " ".join(context.args)
        user_id = update.effective_user.id

        try:
            await update.message.reply_text("🤔 Ищу ответ в документах...")

            # Получаем ответ через RAG пайплайн
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.rag_pipeline.ask_question, question, user_id
            )

            # Форматируем ответ с источниками
            answer_text = f"**Ответ:** {response['answer']}\n\n"

            if response['sources']:
                answer_text += "**Источники:**\n"
                for i, source in enumerate(response['sources'][:3], 1):
                    answer_text += f"{i}. {source['source']} (сходство: {source['score']:.2f})\n"

            await update.message.reply_text(answer_text)

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            await update.message.reply_text("❌ Ошибка при поиске ответа")

    async def get_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        try:
            stats = self.rag_pipeline.get_collection_stats()
            stats_text = (
                f"📊 Статистика базы знаний:\n"
                f"• Всего чанков: {stats['total_chunks']}\n"
                f"• Всего документов: {stats['total_documents']}\n"
                f"• Размер коллекции: {stats['collection_size']} векторов\n"
                f"• Модель эмбеддингов: {stats['embedding_model']}\n"
                f"• LLM модель: {stats['llm_model']}"
            )
            await update.message.reply_text(stats_text)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            await update.message.reply_text("❌ Ошибка при получении статистики")

    async def reset_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /reset"""
        user_id = update.effective_user.id

        # Очищаем сессию пользователя
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]

        # Очищаем историю чата
        if user_id in self.chat_histories:
            del self.chat_histories[user_id]

        await update.message.reply_text("✅ Сессия очищена! История поиска и чата сброшена.")

async def handle_webhook(request):
    """Обработчик вебхуков для сервера"""
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return web.Response(text="OK")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return web.Response(text="ERROR", status=500)

def main():
    """Запуск бота"""
    global application
    bot = RAGTelegramBot()

    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("add_docs", bot.add_documents))
    application.add_handler(CommandHandler("ask", bot.ask_question))
    application.add_handler(CommandHandler("stats", bot.get_stats))
    application.add_handler(CommandHandler("reset", bot.reset_session))
    application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text_message))

    # Проверяем режим запуска
    webhook_url = os.getenv('WEBHOOK_URL')
    webhook_port = int(os.getenv('WEBHOOK_PORT', '8443'))

    if webhook_url:
        # Режим вебхуков для сервера
        print(f"🤖 RAG Telegram Bot запущен в режиме вебхуков на порту {webhook_port}...")

        # Устанавливаем вебхук
        webhook_url = f"{webhook_url}/webhook"
        application.bot.set_webhook(url=webhook_url)

        # Создаем веб-сервер для обработки вебхуков
        app = web.Application()
        app.router.add_post('/webhook', handle_webhook)

        # Запускаем сервер
        web.run_app(app, port=webhook_port, access_log=None)
    else:
        # Режим polling для локальной разработки
        print("🤖 RAG Telegram Bot запущен в режиме polling...")
        application.run_polling()

if __name__ == "__main__":
    main()