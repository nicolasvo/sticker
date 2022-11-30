import json
import os
import random

import boto3
from telegram import Bot

from emojis import emoji_number, get_animated_emojis

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)


def handler(event, context):
    sqs = boto3.resource("sqs")
    queue = sqs.get_queue_by_name(
        QueueName=os.getenv("QUEUE_NAME"),
    )
    update = json.dumps(event["body"])
    response = queue.send_message(MessageBody=update)
    print(response)
    body = json.loads(event["body"])
    if body.get("callback_query"):
        message_id = body["callback_query"]["message"]["message_id"]
        reply_message_id = body["callback_query"]["message"]["reply_to_message"][
            "message_id"
        ]
        chat_id = body["callback_query"]["message"]["chat"]["id"]
        reply_data = body["callback_query"]["data"]
        bot.edit_message_text(
            message_id=message_id,
            chat_id=chat_id,
            text=f"You chose {emoji_number(int(reply_data))}",
            reply_markup=None,
        )
        bot.send_message(
            chat_id,
            f"{random.choice(get_animated_emojis())}",
            reply_to_message_id=reply_message_id,
        )

    return 200
