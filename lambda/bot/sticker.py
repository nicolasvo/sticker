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
        output_path = f"{tmpdirname}/output.png"
        postprocessing(
            image_path, output_path, masks, boxes, tmpdirname, white_outline=False
        )
        image_url = upload_file_and_get_presigned_url(
            bucket_images, f"{tmpdirname}.png", output_path
        )
        upsert_item(
            user_id, segmented_photo=image_url
        )  # TODO: store presigned url, check download speed

    keyboard = [
        [
            InlineKeyboardButton("yes", callback_data="yes"),
            InlineKeyboardButton("no", callback_data="no"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await bot.send_message(
        chat_id=update.effective_chat.id,
        reply_to_message_id=str(message_id),
        reply_markup=reply_markup,
        text="Do you want to make a sticker? ✨",
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

    return masks, boxes


# import os
# import tempfile
# from datetime import datetime

# from telegram import (
#     Bot,
#     Update,
#     InputMediaPhoto,
#     InlineKeyboardButton,
#     InlineKeyboardMarkup,
# )

# import cv2
# import numpy as np
# from PIL import Image

# from lang_sam import LangSAM
# from lang_sam.utils import draw_image

# from emojis import emoji_number
# from image import rescale_img, load_img, write_img, add_outline
# from user import User

# BOT_API_TOKEN = os.getenv("BOT_API_TOKEN")
# bot = Bot(BOT_API_TOKEN)


# def segment_photo2(update: Update) -> None:
#     photo_id = update.message.photo[-1].file_id
#     with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
#         photo_path = f"{tmpdirname}/{photo_id}.jpeg"
#         photo_file = update.message.reply_to_message.photo[-1].file_id
#         photo_file = bot.getFile(photo_file)
#         photo_file.download(photo_file)

#         masks, boxes, _, _ = segment2(photo_path, update.message.caption, tmpdirname)
#         postprocessing(photo_path, masks, boxes, tmpdirname, white_outline=False)

#         output_path = f"{tmpdirname}/output.png"  # TODO: dev ugly

#         keyboard = [
#             [
#                 InlineKeyboardButton("yes", callback_data="yes"),
#                 InlineKeyboardButton("no", callback_data="no"),
#             ]
#         ]
#         reply_markup = InlineKeyboardMarkup(keyboard)

#         update.message.reply_photo(
#             open(output_path, "rb"),
#             reply_to_message_id=update.message.message_id,
#             reply_markup=reply_markup,
#             caption="Is this picture oké la? 🤔",
#         )


# def segment2(photo_path, text_prompt, tmpdirname):
#     model = LangSAM()
#     image_pil = Image.open(photo_path).convert("RGB")
#     output_path = f"{tmpdirname}/output_draw.png"
#     masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
#     labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
#     image_array = np.asarray(image_pil)
#     image_draw = image = draw_image(image_array, masks, boxes, labels)
#     image = Image.fromarray(np.uint8(image)).convert("RGB")
#     cv2.imwrite(output_path, cv2.cvtColor(image_draw, cv2.COLOR_RGBA2BGRA))

#     return masks, boxes, phrases, logits


# def postprocessing(photo_path, masks, boxes, tmpdirname, white_outline=True):
#     image = cv2.imread(photo_path)
#     mask_array = masks.numpy()

#     # Ensure the mask and image have the same dimensions
#     mask_array = np.transpose(mask_array, (1, 2, 0))
#     mask_array = mask_array.astype(np.uint8)
#     mask_array = cv2.resize(
#         mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
#     )

#     if len(mask_array.shape) == 2:
#         masks = [(mask_array > 0).astype(np.uint8)]
#     else:
#         masks = [
#             (mask_array[:, :, i] > 0).astype(np.uint8)
#             for i in range(mask_array.shape[2])
#         ]

#     # Combine the binary masks into a single mask
#     mask = np.any(masks, axis=0).astype(np.uint8)

#     # Create a masked image
#     masked_image = cv2.bitwise_and(image, image, mask=mask)

#     # Create a transparent image with an alpha channel
#     output_segmented_path = f"{tmpdirname}/output_segmented.png"
#     output_image = np.concatenate((masked_image, mask[:, :, np.newaxis] * 255), axis=2)
#     cv2.imwrite(output_segmented_path, output_image)

#     # Example usage
#     bounding_boxes = boxes
#     combined_image = crop_image_with_bounding_boxes(
#         output_segmented_path, bounding_boxes
#     )

#     # Save the combined image
#     combined_image_path = f"{tmpdirname}/combined_image.png"
#     combined_image.save(combined_image_path, "PNG")

#     img = cv2.imread(combined_image_path, cv2.IMREAD_UNCHANGED)
#     if white_outline:
#         img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 255, 255),))
#     else:
#         img = add_outline(img, threshold=0, stroke_size=6, colors=((255, 0, 0),))
#     scaled_img = rescale_img(np.array(img))
#     output_path = f"{tmpdirname}/output.png"  # TODO: dev ugly
#     write_img(scaled_img, output_path, alpha=True)

#     print("chatgpt")  # TODO: dev


# def crop_image_with_bounding_boxes(original_image_path, bounding_boxes):
#     # Load the original image
#     image = Image.open(original_image_path).convert("RGBA")

#     # Find the maximum coordinates of the bounding boxes
#     x_min = min(box[0] for box in bounding_boxes)
#     y_min = min(box[1] for box in bounding_boxes)
#     x_max = max(box[2] for box in bounding_boxes)
#     y_max = max(box[3] for box in bounding_boxes)

#     # Calculate the dimensions of the combined image
#     combined_width = int(x_max - x_min)
#     combined_height = int(y_max - y_min)

#     # Create a new image to combine all bounding boxes
#     combined_image = Image.new("RGBA", (combined_width, combined_height), (0, 0, 0, 0))

#     # Iterate through bounding boxes and paste cropped images onto the combined image
#     for box in bounding_boxes:
#         x, y, x_max, y_max = box
#         x = int(x)
#         y = int(y)
#         x_max = int(x_max)
#         y_max = int(y_max)
#         cropped_image = image.crop((x, y, x_max, y_max))
#         combined_image.paste(cropped_image, (int(x - x_min), int(y - y_min)))

#     return combined_image


# def create_sticker(update: Update, segment, reply_data: str) -> None:
#     user = User(update, bot, reply=True)
#     with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
#         print("Reply received")
#         file_id = update.message.reply_to_message.photo[-1].file_id
#         file_path = f"{tmpdirname}/{file_id}.jpeg"
#         out_path = f"{tmpdirname}/{file_id}.png"
#         img_file = bot.getFile(file_id)
#         img_file.download(file_path)
#         print("Photo downloaded")
#         if reply_data == "1":
#             print("Sticker with model 1")
#             segment.boom(file_path, out_path, outline=True)
#         elif reply_data == "2":
#             print("Sticker with model 2")
#             segment_u2net(file_path, out_path)
#         elif reply_data == "3":
#             print("Sticker with model 3")
#             segment_modnet(file_path, out_path)
#         elif reply_data == "4":
#             print("Sticker without model")
#             img = load_img(file_path)
#             img = rescale_img(img)
#             write_img(img, out_path)
#         try:
#             bot.get_sticker_set(user.sticker_set_name)
#             add_sticker(user, out_path)
#             sticker_set = bot.get_sticker_set(user.sticker_set_name)
#             bot.send_sticker(
#                 user.chat_id,
#                 sticker_set.stickers[-1],
#                 reply_to_message_id=update.message.reply_to_message.message_id,
#             )
#         except Exception as e:
#             add_sticker_pack(user, out_path)
#             sticker_set = bot.get_sticker_set(user.sticker_set_name)
#             bot.send_sticker(
#                 user.chat_id,
#                 sticker_set.stickers[-1],
#                 reply_to_message_id=update.message.message_id,
#             )


# def handle_image(update: Update, segment) -> None:
#     user = User(update, bot)
#     with tempfile.TemporaryDirectory(dir="/tmp/") as tmpdirname:
#         if update.message.photo:
#             print("Photo received")
#             file_id = update.message.photo[-1].file_id
#             file_path = f"{tmpdirname}/{file_id}.jpeg"
#             out_path = f"{tmpdirname}/{file_id}.png"
#             img_file = bot.getFile(file_id)
#             img_file.download(file_path)
#             print("Photo downloaded")
#             if update.message.caption and update.message.caption == "/file":
#                 segment.bill(file_path, out_path)
#                 send_file(update, out_path)
#                 return
#             elif update.message.caption and update.message.caption == "/please":
#                 img = load_img(file_path)
#                 img = rescale_img(img)
#                 write_img(img, out_path)
#             elif update.message.caption and update.message.caption == "/no":
#                 segment.boom(file_path, out_path, outline=False)
#                 print("Photo segmented without outline")
#             else:
#                 images = []

#                 out_path = f"{tmpdirname}/xception_{file_id}.png"
#                 segment.boom(file_path, out_path, outline=True, white_outline=False)
#                 images.append(out_path)
#                 print("Photo segmented with model 1")

#                 out_path = f"{tmpdirname}/u2net_{file_id}.png"
#                 segment_u2net(file_path, out_path, white_outline=False)
#                 images.append(out_path)
#                 print("Photo segmented with model 2")

#                 out_path = f"{tmpdirname}/modnet_{file_id}.png"
#                 segment_modnet(file_path, out_path, white_outline=False)
#                 images.append(out_path)
#                 print("Photo segmented with model 3")

#                 out_path = f"{tmpdirname}/please_{file_id}.png"
#                 img = load_img(file_path)
#                 img = rescale_img(img)
#                 write_img(img, out_path)
#                 images.append(out_path)

#             try:
#                 medias = [InputMediaPhoto(open(path, "rb")) for path in images]
#                 keyboard = [
#                     [
#                         InlineKeyboardButton(
#                             emoji_number(n + 1), callback_data=str(n + 1)
#                         )
#                         for n in range(len(images))
#                     ]
#                 ]
#                 reply_markup = InlineKeyboardMarkup(keyboard)
#                 bot.send_media_group(
#                     user.chat_id,
#                     medias,
#                     reply_to_message_id=update.message.message_id,
#                 )
#                 bot.send_message(
#                     chat_id=user.chat_id,
#                     text="Choose a picture 💬",
#                     reply_markup=reply_markup,
#                     reply_to_message_id=update.message.message_id,
#                 )
#             except Exception as e:
#                 print(e)


# def add_sticker(user: User, sticker_path: str) -> None:
#     with open(sticker_path, "rb") as sticker:
#         file = bot.upload_sticker_file(user.id, sticker)
#         bot.add_sticker_to_set(user.id, user.sticker_set_name, "🎨", file.file_id)
#         print("Sticker added")


# def add_sticker_pack(user: User, sticker_path: str) -> None:
#     with open(sticker_path, "rb") as sticker:
#         file = bot.upload_sticker_file(user.id, sticker)
#         bot.create_new_sticker_set(
#             user.id,
#             user.sticker_set_name,
#             user.sticker_set_title,
#             "🎨",
#             file.file_id,
#         )
#         print("Sticker set added")


# def delete_sticker(update: Update) -> None:
#     user = User(update, bot)
#     bot.delete_sticker_from_set(
#         bot.get_sticker_set(user.sticker_set_name).stickers[-1].file_id
#     )
#     print("Sticker deleted")
#     bot.send_message(
#         update.message.chat_id,
#         "Last sticker deleted! 🥲",
#         reply_to_message_id=update.message.message_id,
#     )


# def send_file(update: Update, file_path: str) -> None:
#     bot.send_document(
#         chat_id=update.message.chat_id,
#         document=open(file_path, "rb"),
#         filename=f"photo_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.png",
#     )
