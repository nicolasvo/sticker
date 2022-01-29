import os
from telegram import Bot, Update, ChatAction


BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")


def handler(event, context):
    print(f"event: {type(event)}, {event}")
    print(f"context: {context}")
    bot = Bot(BOT_API_TOKEN)
    update = Update.de_json(event.body, bot)
    bot.send_message(update.message.chat_id, f"lambda: {update.message.text}")

    return 200
