# 🤖 RAG Telegram Bot

[![CI](https://github.com/MrAlexGov/RAG-Telegram-Bot/actions/workflows/ci.yml/badge.svg)](https://github.com/MrAlexGov/RAG-Telegram-Bot/actions/workflows/ci.yml)

AI-ассистент в Telegram с локальными моделями через Ollama. Использует Retrieval-Augmented Generation (RAG) архитектуру для обработки документов и ответа на вопросы.

## 📋 Описание

Бот позволяет:
- Загружать и обрабатывать документы (PDF, TXT, DOCX)
- Строить векторную базу знаний
- Отвечать на вопросы по загруженным документам
- Цитировать источники в ответах
- Использовать локальные модели без облачных сервисов

## 🛠 Технический стек

- **Python 3.9+**
- **Telegram Bot API** (python-telegram-bot v20+)
- **Ollama** (локальные модели)
- **Qdrant** (векторная база данных)
- **LangChain** (обработка документов)
- **Sentence Transformers** (эмбеддинги)

## 🚀 Быстрый старт

### 1. Установка Ollama

Скачайте и установите Ollama с [официального сайта](https://ollama.ai):

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - скачайте с сайта
```

### 2. Загрузка моделей

```bash
# Основная модель для генерации ответов
ollama pull qwen2:7b

# Модель для создания эмбеддингов
ollama pull nomic-embed-text
```

### 3. Клонирование и настройка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd rag-telegram-bot

# Запустите скрипт настройки
python setup.py
```

### 4. Настройка Telegram бота

1. Создайте бота через [@BotFather](https://t.me/botfather) в Telegram
2. Скопируйте токен бота
3. Создайте файл `.env` в корне проекта:

```env
TELEGRAM_BOT_TOKEN=ваш_токен_здесь
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=:memory:
```

### 5. Запуск бота

```bash
python bot.py
```

## 📖 Использование

### Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие и инструкция по использованию |
| `/add_docs` | Запрос на добавление документов |
| `/ask [вопрос]` | Задать вопрос по загруженным документам |
| `/stats` | Показать статистику базы знаний |
| `/reset` | Очистить текущую сессию |

### Работа с ботом

1. **Добавление документов:**
   - Отправьте боту файл (PDF, TXT или DOCX)
   - Или используйте команду `/add_docs` для получения инструкций
   - Бот автоматически обработает документ и добавит в базу знаний

2. **Вопросы по документам:**
   - Используйте команду `/ask Ваш вопрос здесь`
   - Бот найдет релевантную информацию и сгенерирует ответ
   - В ответе будут указаны источники с ссылками на документы

3. **Проверка статистики:**
   - Команда `/stats` покажет количество чанков, документов и размер базы

### Пример использования

```
Пользователь: /start
Бот: 🤖 Привет! Я RAG-бот с локальными моделями...

Пользователь: [отправляет PDF файл]
Бот: 📥 Документ 'manual.pdf' получен. Начинаю обработку...
✅ Документ обработан!
• Извлечено чанков: 45
• Размер базы: 145 чанков

Пользователь: /ask Как настроить систему?
Бот: **Ответ:**
Система настраивается через конфигурационный файл...

**Источники:**
1. manual.pdf (сходство: 0.92)
2. manual.pdf (сходство: 0.85)
```

## 📁 Структура проекта

```
rag-telegram-bot/
├── bot.py                 # Основной файл Telegram бота
├── rag_pipeline.py        # RAG пайплайн и обработка документов
├── config.py             # Конфигурация и переменные окружения
├── setup.py              # Скрипт автоматической настройки
├── requirements.txt      # Зависимости Python
├── .env                  # Переменные окружения (создать самостоятельно)
├── data/                 # Папка для документов (создается автоматически)
├── temp/                 # Временные файлы (создается автоматически)
├── logs/                 # Логи приложения (создается автоматически)
└── README.md             # Этот файл
```

## ⚙️ Конфигурация

### Переменные окружения (.env)

```env
# Обязательные
TELEGRAM_BOT_TOKEN=ваш_telegram_bot_token

# Опциональные
OLLAMA_BASE_URL=http://localhost:11434    # URL Ollama сервера
QDRANT_URL=:memory:                       # URL Qdrant (или файловая БД)
QDRANT_API_KEY=                           # API ключ Qdrant (если нужен)
```

### Настройки в config.py

```python
EMBEDDING_MODEL = "nomic-embed-text"      # Модель эмбеддингов
LLM_MODEL = "qwen2:7b"                   # Модель генерации
CHUNK_SIZE = 1000                        # Размер чанка текста
CHUNK_OVERLAP = 200                      # Перекрытие чанков
MAX_TOKENS = 4000                        # Максимум токенов в ответе
```

## 🔧 Расширенная настройка

### Использование Qdrant сервера

Для продакшена рекомендуется использовать отдельный Qdrant сервер:

```bash
# Установка Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Настройка в .env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=ваш_api_key  # если настроена аутентификация
```

### Кастомные модели

Измените модели в `config.py`:

```python
# Другие модели для эмбеддингов
EMBEDDING_MODEL = "mxbai-embed-large"

# Другие LLM модели
LLM_MODEL = "llama2:13b"
LLM_MODEL = "mistral:7b"
```

### Обработка больших документов

Для больших PDF файлов (>100 страниц) бот автоматически:
- Разделяет документ на чанки
- Обрабатывает чанки параллельно
- Кэширует эмбеддинги для ускорения

## 🐛 Устранение неполадок

### Проблемы с Ollama

```bash
# Проверить статус Ollama
ollama list

# Перезапустить сервис
sudo systemctl restart ollama

# Проверить доступность API
curl http://localhost:11434/api/tags
```

### Ошибки при установке зависимостей

```bash
# Очистить кэш pip
pip cache purge

# Установить в виртуальном окружении
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Бот не отвечает

1. Проверьте токен бота в `.env`
2. Убедитесь, что бот запущен: `python bot.py`
3. Проверьте логи на ошибки
4. Перезапустите бота

### Ошибки обработки документов

- Поддерживаемые форматы: PDF, TXT, DOCX
- Максимальный размер файла: зависит от Telegram (обычно 20MB)
- Для больших файлов используйте сжатие или разделение

## 📊 Мониторинг и логи

Бот ведет логи в папке `logs/`:
- `bot.log` - основные события бота
- `rag.log` - логи RAG пайплайна
- `errors.log` - ошибки и исключения

### Просмотр логов

```bash
# Последние 50 строк логов
tail -50 logs/bot.log

# Мониторинг в реальном времени
tail -f logs/bot.log
```

## 🌐 Настройка вебхуков для публичного доступа

Если бот запускается на локальной машине и вам нужен публичный доступ, настройте вебхуки с ngrok:

### 1. Установка ngrok

```bash
# Автоматическая установка
./webhook_setup.sh

# Или вручную
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
```

### 2. Настройка токена ngrok

```bash
# Получите токен на https://dashboard.ngrok.com/get-started/your-authtoken
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### 3. Запуск туннеля

```bash
# В одном терминале запускаем ngrok
ngrok http 8443

# В другом терминале запускаем бота
python bot.py
```

### 4. Настройка .env файла

```env
# Добавьте в .env
WEBHOOK_URL=https://your-ngrok-url.ngrok.io
WEBHOOK_PORT=8443
```

### 5. Перезапуск бота

После настройки перезапустите бота - он автоматически переключится в режим вебхуков.

## 🔒 Безопасность

- Все данные хранятся локально
- Нет передачи данных в облако
- Используйте сильные токены для бота
- Ngrok токен храните в секрете
- Регулярно обновляйте зависимости

## 🤝 Contributing

1. Fork репозиторий
2. Создайте feature branch: `git checkout -b feature/new-feature`
3. Commit изменения: `git commit -am 'Add new feature'`
4. Push в branch: `git push origin feature/new-feature`
5. Создайте Pull Request

## 🧪 Тесты

Юнит-тесты для конфигурации + проверка синтаксиса всех модулей (без необходимости
разворачивать Ollama/Qdrant в CI):

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей.

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте логи в папке `logs/`
2. Создайте issue в репозитории
3. Опишите проблему с шагами воспроизведения

## 🔄 Обновления

Для обновления проекта:

```bash
git pull
python setup.py
```

### Changelog

- **v1.0.0**: Первый релиз с базовым функционалом RAG
- Поддержка PDF, TXT, DOCX
- Интеграция с Ollama и Qdrant
- Telegram интерфейс

---

*Создано с ❤️ для локального AI*