import os
import json
import random
from telegram import Bot, Update, ChatAction
from telegram.error import TelegramError

from alter_background import AlterBackground
from sticker import create_sticker, delete_sticker
from emojis import get_emojis, get_animated_emojis

segment = AlterBackground(model_type="pb")
segment.load_pascalvoc_model("xception_pascalvoc.pb")

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)


def handler(event, context):
    try:
        update = Update.de_json(json.loads(event["body"]), bot)
        if update.message.photo or update.message.document:
            bot.send_message(
                update.message.chat_id,
                f"{random.choice(get_animated_emojis())}",
                reply_to_message_id=update.message.message_id,
            )
            create_sticker(update, segment)
        elif update.message.text:
            if update.message.text == '/delete':
                delete_sticker(update)
        else:
            bot.send_message(update.message.chat_id, f"Send me a picture! {random.choice(get_emojis())}")

        return 200
    except Exception as e:
        print(f"Error: {e}")
        bot.send_message(
            update.message.chat_id, f"🧨", reply_to_message_id=update.message.message_id
        )
        return 500
