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

from dynamodb import get_item


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
    await bot.send_message(
        chat_id=update.effective_chat.id,
        reply_to_message_id=str(message_id),
        text="Segmented this photo",
    )
