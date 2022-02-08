import os
import json
import random
from telegram import Bot, Update, ChatAction

from emojis import get_emojis

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)


def handler(event, context):
    update = Update.de_json(json.loads(event["body"]), bot)
    bot.send_message(update.message.chat_id, f"Sorry, I'm down at the moment... {random.choice(get_emojis())}")

    return 200

