import json
import os
import random

from telegram import Bot, Update

from alter_background import AlterBackground
from emojis import get_emojis
from sticker import create_sticker, delete_sticker, handle_image

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)


def handler(event, context):
    try:
        process_image = False
        process_sticker = False
        _update = json.loads(json.loads(event["Records"][0]["body"]))
        update = Update.de_json(_update, bot)
        raw = event["Records"][0]["body"]
        reply = json.loads(json.loads(raw))
        if not reply.get("callback_query"):
            if update.message.photo or update.message.document:
                process_image = True
            elif update.message.text and update.message.text == "/delete":
                delete_sticker(update)
            else:
                bot.send_message(
                    update.message.chat_id,
                    f"Send me a picture! {random.choice(get_emojis())}",
                )
        else:
            raw = event["Records"][0]["body"]
            reply = json.loads(json.loads(raw))
            reply_data = reply["callback_query"]["data"]
            _update = {
                "update_id": reply["update_id"],
                "message": reply["callback_query"]["message"],
            }
            update = Update.de_json(_update, bot)
            process_image = True
            process_sticker = True
        if process_image:
            if "segment" in globals():
                print("Model already loaded")
            else:
                print("Loading model")
                global segment
                segment = AlterBackground(model_type="pb")
                segment.load_pascalvoc_model("xception_pascalvoc.pb")
            if process_sticker:
                create_sticker(update, segment, reply_data)
            else:
                handle_image(update, segment)

        return 200
    except Exception as e:
        print(f"Error: {e}")
        bot.send_message(
            update.message.chat_id, f"🧨", reply_to_message_id=update.message.message_id
        )
        return 500
