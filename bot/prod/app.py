import os
import json
import tempfile
from telegram import Bot, Update, ChatAction
from telegram.error import TelegramError

from alter_background import AlterBackground
from sticker import create_sticker

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
                f"Creating sticker 🪄",
                reply_to_message_id=update.message.message_id,
            )
            create_sticker(update)
        else:
            bot.send_message(update.message.chat_id, f"lambdo {update.message.text}")

        return 200
    except Exception as e:
        print(f"Error: {e}")
        bot.send_message(
            update.message.chat_id, f"🧨", reply_to_message_id=update.message.message_id
        )
        return 500
