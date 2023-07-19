import json
import os
import random

import boto3
from telegram import Bot, Update

from emojis import get_emojis
from sticker import delete_sticker, segment_photo2
from dynamodb import get_item

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)
dynamodb = boto3.resource("dynamodb")


def handler(event, context):
    try:
        _update = json.loads(json.loads(event["Records"][0]["body"]))
        update = Update.de_json(_update, bot)
        raw = event["Records"][0]["body"]
        reply = json.loads(json.loads(raw))
        if not reply.get("callback_query"):
            user_id = update.effective_user.id
            print(f"user id: {user_id}")
            item = get_item(user_id)
            if item:
                print(f"Record: {item}")
            else:  # TODO: dev debug
                print(f"Record does not exist for user {user_id}")

            # new photo
            if update.message.photo or update.message.document:
                print("User sent a new photo")
                segment_photo2(update)

            # delete sticker
            elif update.message.text and update.message.text == "/delete":
                delete_sticker(update)

            # new prompt
            elif update.message.text and item and item.get("FileId"):
                print("User sent a new prompt")
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

        return 200
    except Exception as e:
        print(f"Error: {e}")
        bot.send_message(
            update.message.chat_id, f"🧨", reply_to_message_id=update.message.message_id
        )
        return 500
