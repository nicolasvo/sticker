import os
import json
import logging
import tempfile
from datetime import datetime
from telegram import Bot, Update, ChatAction
from telegram.error import TelegramError

from image import rescale_img, load_img, write_img
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
            if update.message.caption and update.message.caption == "/file":
                segment.bill(file_path, out_path)
                send_file(update, out_path)
                return
            elif update.message.caption and update.message.caption == "/please":
                img = load_img(file_path)
                img = rescale_img(img)
                write_img(img, out_path)
            elif update.message.caption and update.message.caption == "/no":
                segment.boom(file_path, out_path, outline=False)
                print("Photo segmented without outline")
            else:
                segment.boom(file_path, out_path, outline=True)
                print("Photo segmented with outline")

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
        bot.add_sticker_to_set(user.id, user.sticker_set_name, "🎨", file.file_id)
        print("Sticker added to set")


def add_sticker_pack(user: User, sticker_path: str) -> None:
    with open(sticker_path, "rb") as sticker:
        file = bot.upload_sticker_file(user.id, sticker)
        bot.create_new_sticker_set(
            user.id,
            user.sticker_set_name,
            user.sticker_set_title,
            "🎨",
            file.file_id,
        )
        print("Sticker set added")


def delete_sticker(update: Update) -> None:
    user = User(update, bot)
    bot.delete_sticker_from_set(
        bot.get_sticker_set(user.sticker_set_name).stickers[-1].file_id
    )
    print("Sticker deleted")
    bot.send_message(
        update.message.chat_id,
        "Last sticker deleted! 🥲",
        reply_to_message_id=update.message.message_id,
    )


def send_file(update: Update, file_path: str) -> None:
    bot.send_document(
        chat_id=update.message.chat_id,
        document=open(file_path, "rb"),
        filename=f"photo_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.png",
    )