import asyncio
import json
import os

from telegram import Bot, Update

loop = asyncio.get_event_loop()


async def main(event, context):
    try:
        BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
        bot = Bot(BOT_API_TOKEN)
        update = json.loads(event["body"])
        update = Update.de_json(update, bot)
        print(update)
        await bot.send_message(
            update.message.chat_id,
            f"Send me a picture la! {update.message.text}",
        )

        user_id = update.message.from_user.id
        # new photo

        # new prompt

        # delete # send the sticker you want to delete

        # anything else
    

    except Exception as e:
        print(e)
        raise e
    finally:
        return {
            "statusCode": 200,
        }


def lambda_handler(event, context):
    return loop.run_until_complete(main(event, context))
