import os
import json
import logging
from telegram import Bot, Update, ChatAction

from alter_background import AlterBackground


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

segment = AlterBackground(model_type="pb")
segment.load_pascalvoc_model("xception_pascalvoc.pb")

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)

def handler(event, context):
    update = Update.de_json(json.loads(event["body"]), bot)
    if update.message.photo or update.message.document:
        make_sticker(update)
    else:
        bot.send_message(update.message.chat_id, f"lambdo {update.message.text}")

    return 200

def make_sticker(update: Update) -> None:
    chat_id = update.message.chat_id
    if update.message.photo:
        logger.info("Photo received")
        file_id = update.message.photo[-1].file_id  
        file_path = "/tmp/test_photo.jpeg"
        out_path = "/tmp/test_photo.png"
        image_file = bot.getFile(file_id)
        image_file.download(file_path)
        logger.info("Photo downloaded")
        segment.boom(file_path, out_path, "person")
        logger.info("Photo segmented")
        bot.send_document(chat_id, open(out_path, 'rb'))
        logger.info("Photo sent")
