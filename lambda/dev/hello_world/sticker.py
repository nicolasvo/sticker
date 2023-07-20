import os
import tempfile

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


async def segment_photo2(update: Update) -> None:
    BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
    bot = Bot(BOT_API_TOKEN)
    await bot.initialize()
    await update.message.reply_text("Starto")
    print("Starting segmentation")
    user_id = update.effective_user.id
    item = get_item(user_id)
    photo_id = item.get("FileId")
    message_id = item.get("MessageId")
    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        photo_path = f"{tmpdirname}/{photo_id}.jpeg"
        photo_file = await bot.get_file(photo_id)
        await photo_file.download_to_drive(custom_path=photo_path)
        print("Photo downloaded")

    keyboard = [
        [
            InlineKeyboardButton("yes", callback_data="yes"),
            InlineKeyboardButton("no", callback_data="no"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # print("Storing base64 image into table item")
    # upsert_item(user_id, segmented_photo=image_string)

    await bot.send_message(
        chat_id=update.effective_chat.id,
        reply_to_message_id=str(message_id),
        reply_markup=reply_markup,
        text="Segmented this photo",
    )


async def make_sticker2(update: Update) -> None:
    # BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
    # bot = Bot(BOT_API_TOKEN)
    # await bot.initialize()
    print("Making sticker")
    item = get_item(update.effective_user.id)
    # segmented_photo = item["SegmentedPhoto"]
    with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
        output_path = f"{tmpdirname}/output.png"
        # base64_to_image(segmented_photo, output_path)


def image_to_base64(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


def base64_to_image(base64_string, output_file_path):
    import base64

    with open(output_file_path, "wb") as image_file:
        decoded_image = base64.b64decode(base64_string)
        image_file.write(decoded_image)
