import asyncio
import json
import os
import requests

import boto3
from telegram import Bot, Update

from dynamodb import get_item, upsert_item
from sticker import request_segment

loop = asyncio.get_event_loop()


def trigger_lambda(event, context):
    client = boto3.client("lambda")
    response = client.invoke(
        FunctionName="sticker-sam",
        InvocationType="Event",  # Asynchronous invocation
        Payload=json.dumps(
            {
                "bizarre": {"body": event},
            }
        ),  # Payload can be a JSON object or any data to pass to the target Lambda
    )


async def main(event, context):
    try:
        BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
        bot = Bot(BOT_API_TOKEN)
        print(f"event: {event}")
        if event.get("bizarre"):
            print("bizarre shit")
            print(f"event: {event}")
            print(f"body: {event['bizarre']['body']}")
            update = json.loads(event["bizarre"]["body"])
            update = Update.de_json(update, bot)
            await request_segment(update, update.message.text)
            return {
                "statusCode": 200,
            }
        else:
            print(f"body: {event['body']}")
            update = json.loads(event["body"])
            update = Update.de_json(update, bot)
            print(update)

        if not update.callback_query:
            user_id = update.effective_user.id
            item = get_item(user_id)

            # delete
            # send the sticker you want to delete

            # new photo
            if update.message.photo or update.message.document:
                print("User sent a new photo")
                upsert_item(
                    user_id, update.message.id, update.message.photo[-1].file_id, None
                )
                await bot.send_message(
                    update.message.chat_id,
                    f"Write what to cut from the picture 💬\nFor example: person left and chocolate cake",
                )

            # new prompt
            elif update.message.text and item and item.get("FileId"):
                print(f"User sent a new prompt: {update.message.text}")
                await update.message.reply_text("Analyzing picture 🧠")
                print("hoe")
                trigger_lambda(event["body"], context)
                print("tram")
                # finally:
                #     # await request_segment(update, update.message.text)
                #     print("gratteur")

            # anything else
            else:
                await bot.send_message(
                    update.message.chat_id,
                    f"Send me a picture la!",
                )
        else:
            # user confirms sticker
            answer = update.callback_query.data
            if answer == "yes":
                print("User confirmed sticker creation")

    except Exception as e:
        print(e)
        raise e
    finally:
        return {
            "statusCode": 200,
        }


def lambda_handler(event, context):
    return loop.run_until_complete(main(event, context))
