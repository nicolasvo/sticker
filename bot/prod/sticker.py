import os
import json
import logging
import tempfile
from telegram import Bot, Update, ChatAction
from telegram.error import TelegramError

from alter_background import AlterBackground
from user import User

BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
bot = Bot(BOT_API_TOKEN)


def create_sticker(update: Update, segment) -> None:
    user = User(update, bot)
    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        if update.message.photo:
            print("Photo received")
            file_id = update.message.photo[-1].file_id
            file_path = f"{tmpdirname}/{file_id}.jpeg"
            out_path = f"{tmpdirname}/{file_id}.png"
            img_file = bot.getFile(file_id)
            img_file.download(file_path)
            print("Photo downloaded")
            segment.boom(file_path, out_path, "person")
            print("Photo segmented")
            # bot.send_document(chat_id, open(out_path, "rb"))
            # print("Photo sent")

            try:
                bot.get_sticker_set(user.sticker_set_name)
                add_sticker(user, out_path)
                sticker_set = bot.get_sticker_set(user.sticker_set_name)
                bot.send_sticker(
                    user.chat_id,
                    sticker_set.stickers[-1],
                    reply_to_message_id=update.message.message_id,
                )
            except:
                add_sticker_pack(user, out_path)
                sticker_set = bot.get_sticker_set(user.sticker_set_name)
                bot.send_sticker(
                    user.chat_id,
                    sticker_set.stickers[-1],
                    reply_to_message_id=update.message.message_id,
                )


def add_sticker(user: User, sticker_path: str) -> None:
    with open(sticker_path, "rb") as sticker:
        file = bot.upload_sticker_file(user.id, sticker)
        bot.add_sticker_to_set(user.id, user.sticker_set_name, "💊", file.file_id)
        print("Sticker added to set")


def add_sticker_pack(user: User, sticker_path: str) -> None:
    with open(sticker_path, "rb") as sticker:
        file = bot.upload_sticker_file(user.id, sticker)
        bot.create_new_sticker_set(
            user.id,
            user.sticker_set_name,
            f"{user.firstname}land",
            "💊",
            file.file_id,
        )
        print("Sticker set added")