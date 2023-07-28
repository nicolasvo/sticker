import base64
import json
import os
import requests
import tempfile

import numpy as np
import torch
import zlib

from telegram import (
    Bot,
    Update,
    InputMediaPhoto,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from telegram.ext import (
    ContextTypes,
    CallbackContext,
)

from dynamodb import get_item, delete_item, upsert_item
from image import image_to_base64, postprocessing, upload_file_and_get_presigned_url
from user import User

segment_sam_url = os.getenv("SEGMENT_SAM_URL")
bucket_images = os.getenv("BUCKET_IMAGES")


async def request_segment(update: Update, text_prompt=None) -> None:
    BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
    bot = Bot(BOT_API_TOKEN)
    await bot.initialize()
    user_id = update.effective_user.id
    item = get_item(user_id)
    photo_id = item.get("FileId")
    message_id = item.get("MessageId")

    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        image_path = f"{tmpdirname}/{photo_id}.jpeg"
        image_file = await bot.get_file(photo_id)
        await image_file.download_to_drive(custom_path=image_path)
        print("Photo downloaded")
        # TODO: upload image to segment-images, store file id into record
        image = image_to_base64(image_path)
        masks, boxes = request_segment_(image, text_prompt, segment_sam_url)
        output_path_ask = f"{tmpdirname}/output_ask.png"
        output_path = f"{tmpdirname}/output.png"
        postprocessing(
            image_path, output_path_ask, masks, boxes, tmpdirname, white_outline=False
        )
        postprocessing(
            image_path, output_path, masks, boxes, tmpdirname, white_outline=True
        )
        print("done preprocessing")
        image_url = upload_file_and_get_presigned_url(
            bucket_images, f"{tmpdirname}.png", output_path
        )
        print("done uploading to s3")
        upsert_item(user_id, segmented_photo=image_url)

        keyboard = [
            [
                InlineKeyboardButton("yes", callback_data="yes"),
                InlineKeyboardButton("no", callback_data="no"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=open(output_path_ask, "rb"),
            reply_to_message_id=str(message_id),
            reply_markup=reply_markup,
            caption="Do you want to make a sticker? ✨",
        )


def request_segment_(image, text_prompt, lambda_url):
    # TODO: convert and to .
    print("Requesting segmentation")
    payload = {"image": image, "text_prompt": text_prompt}
    url = lambda_url
    r = requests.post(url, json=payload, timeout=600)
    c = r.content
    j = json.loads(c)
    boxes = j["boxes"]
    received_compressed_np = base64.b64decode(j["masks"])
    masks_shape = j["masks_shape"]
    decompressed_np = np.frombuffer(zlib.decompress(received_compressed_np), dtype=bool)
    masks = torch.tensor(decompressed_np.reshape(masks_shape))
    print("donezone")

    return masks, boxes


async def make_sticker(update: Update) -> None:
    BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
    bot = Bot(BOT_API_TOKEN)
    await bot.initialize()
    user = User(update, bot)
    user_id = update.effective_user.id
    item = get_item(user_id)
    segmented_photo = item.get("SegmentedPhoto")
    message_id = item.get("MessageId")
    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        output_path = f"{tmpdirname}.png"
        r = requests.get(segmented_photo)
        with open(output_path, "wb") as file:
            file.write(r.content)
        try:
            bot.get_sticker_set(user.sticker_set_name)
            await add_sticker(user, bot, output_path)
            sticker_set = await bot.get_sticker_set(user.sticker_set_name)
            await bot.send_sticker(
                user.chat_id,
                sticker_set.stickers[-1],
                reply_to_message_id=message_id,
            )
        except Exception as e:
            await add_sticker_pack(user, bot, output_path)
            print("get sticker")
            sticker_set = await bot.get_sticker_set(user.sticker_set_name)
            print("send sticker")
            await bot.send_sticker(
                user.chat_id,
                sticker_set.stickers[-1],
                reply_to_message_id=int(message_id),
            )


async def add_sticker_pack(user: User, bot, sticker_path: str) -> None:
    with open(sticker_path, "rb") as sticker:
        file = await bot.upload_sticker_file(user.id, sticker)
        await bot.create_new_sticker_set(
            user.id,
            user.sticker_set_name,
            user.sticker_set_title,
            "🎨",
            file.file_id,
        )
        print("Sticker set created")


async def add_sticker(user: User, bot, sticker_path: str) -> None:
    with open(sticker_path, "rb") as sticker:
        file = await bot.upload_sticker_file(user.id, sticker)
        await bot.add_sticker_to_set(user.id, user.sticker_set_name, "🎨", file.file_id)
        print("Sticker added")


async def delete_sticker(update: Update) -> None:
    BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
    bot = Bot(BOT_API_TOKEN)
    await bot.initialize()
    user = User(update, bot)

    sticker_set = await bot.get_sticker_set(user.sticker_set_name)
    last_sticker = sticker_set.stickers[-1].file_id
    await bot.delete_sticker_from_set(last_sticker)
    print("Sticker deleted")
    await update.message.reply_text("Last sticker deleted! 🥲")


# def send_file(update: Update, file_path: str) -> None:
#     bot.send_document(
#         chat_id=update.message.chat_id,
#         document=open(file_path, "rb"),
#         filename=f"photo_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.png",
#     )
