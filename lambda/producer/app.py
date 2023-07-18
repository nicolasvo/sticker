import json
import os
import random

import boto3
from telegram import Bot

from emojis import get_animated_emojis

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
        print(f"callback query: {reply_data }")
        bot.edit_message_text(
            message_id=message_id,
            chat_id=chat_id,
            text=f"You chose {reply_data}",
        )
    
    # dynamodb = boto3.resource('dynamodb')

    # table = dynamodb.Table("sticker-sam")
    # response = table.get_item(Key={'UserId': '444'})
    # item = response['Item']
    # print(f"i read: {item}")

    # item = {
    #     "UserId": {"N": "123"},
    #     "FileId": {"S": "something"},
    #     "SegmentedPhoto": {"S": "bobo"},
    # }
    # response = dynamodb.put_item(
    #     TableName="sticker-sam", Item=item  # Replace with your actual table name
    # )

    return 200
