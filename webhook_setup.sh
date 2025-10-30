#!/bin/bash

# Скрипт настройки вебхуков для RAG Telegram бота

echo "🚀 Настройка вебхуков для RAG Telegram бота"

# Проверяем наличие ngrok
if ! command -v ngrok &> /dev/null; then
    echo "❌ Ngrok не установлен. Установите его сначала:"
    echo "curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
    echo "echo \"deb https://ngrok-agent.s3.amazonaws.com buster main\" | sudo tee /etc/apt/sources.list.d/ngrok.list"
    echo "sudo apt update && sudo apt install ngrok"
    exit 1
fi

# Запрашиваем токен ngrok
echo "📝 Получите ваш токен ngrok на https://dashboard.ngrok.com/get-started/your-authtoken"
read -p "Введите ваш ngrok токен: " NGROK_TOKEN

# Авторизуемся в ngrok
ngrok config add-authtoken $NGROK_TOKEN

# Запускаем ngrok на порту 8443
echo "🌐 Запускаем ngrok туннель на порту 8443..."
ngrok http 8443

# Получаем URL вебхука
echo ""
echo "✅ Ngrok запущен!"
echo "📋 Скопируйте HTTPS URL из вывода выше и добавьте в .env файл:"
echo "WEBHOOK_URL=https://your-ngrok-url.ngrok.io"
echo "WEBHOOK_PORT=8443"
echo ""
echo "📝 Пример:"
echo "WEBHOOK_URL=https://abcd1234.ngrok.io"
echo "WEBHOOK_PORT=8443"
echo ""
echo "🔄 После настройки перезапустите бота: python bot.py"