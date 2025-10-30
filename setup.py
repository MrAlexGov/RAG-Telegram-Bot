#!/usr/bin/env python3
"""
Скрипт настройки RAG бота
"""

import subprocess
import sys

def install_requirements():
    """Установка зависимостей"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_ollama():
    """Проверка установки Ollama"""
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        print("✅ Ollama подключен")

        # Проверяем необходимые модели
        required_models = ["qwen2:7b", "nomic-embed-text"]
        available_models = [model['name'] for model in models['models']]

        for model in required_models:
            if model in available_models:
                print(f"✅ Модель {model} доступна")
            else:
                print(f"❌ Модель {model} не найдена. Загрузите: ollama pull {model}")

    except Exception as e:
        print(f"❌ Ошибка подключения к Ollama: {e}")

def create_directories():
    """Создание необходимых директорий"""
    import os
    directories = ['data', 'temp', 'logs']
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        print(f"✅ Создана директория: {dir}")

if __name__ == "__main__":
    print("🚀 Настройка RAG Telegram бота...")

    install_requirements()
    create_directories()
    check_ollama()

    print("\n✅ Настройка завершена!")
    print("📝 Не забудьте:")
    print("1. Создать .env файл с TELEGRAM_BOT_TOKEN")
    print("2. Добавить документы в папку data/")
    print("3. Запустить бота: python bot.py")