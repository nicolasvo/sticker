import logging
import json
import tempfile
import random
import numpy as np
from PIL import Image
from telegram import Bot, Update, ChatAction
from telegram.error import BadRequest, NetworkError
from flask import Flask, request
from app.utils import timeit, emojis
from app.context import Context
from model.generate_labels import generate_labels

with open('app/res/token.json') as f:
    token = json.load(f)['token']

app = Flask(__name__)
bot = Bot(token)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

text_commands = """
Send me a picture or these commands:

/start - 🤖
/help - 🦜
/delete *number* - remove a sticker at a specific position in your set, the whole pack with *all* or the last sticker of your set by not passing a parameter
"""


@app.route('/', methods=['POST'])
def main():
    update = Update.de_json(request.get_json(force=True), bot)
    context = Context(bot, update)

    logger.info(update)

    if update.message.sticker: return 'boi', 200

    bot.send_chat_action(context.chat_id, ChatAction.TYPING)

    if update.message.photo or update.message.document:
        make_sticker(bot, update, context)

    elif update.message.text:
        if update.message.text == '/start':
            text = f"Send me a picture! {random.choice(emojis)}"
            bot.send_message(context.chat_id, text=text)

        elif update.message.text.startswith('/delete'):
            delete_sticker(bot, context, update.message.text)

        elif update.message.text == '/test':
            test(bot, update, context)

        else:
            bot.send_message(context.chat_id, text=text_commands, parse_mode='Markdown')

    else:
        bot.send_message(context.chat_id, text=text_commands, parse_mode='Markdown')

    return 'boi', 200


def make_sticker(bot, update, context):
    # TODO: check if document is a picture
    with tempfile.TemporaryDirectory(dir='/tmp/') as tmpdirname:
        # logger.info('A photo was sent.')
        segmentation = True
        is_png = False
        if update.message.caption:
            if update.message.caption == '/please':
                segmentation = False
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
        elif update.message.document and update.message.document.mime_type.startswith('image'):
            file_id = update.message.document.file_id
            if update.message.document.mime_type.endswith('png'):
                print("It's a png, lads.")
                is_png = True
        else:
            text = 'The file sent was not an image.'
            bot.send_message(context.chat_id, text=text)

            return 'boi', 200

        text = f'Your picture is being made into a sticker! {random.choice(emojis)}'
        bot.send_message(context.chat_id, text=text)
        bot.send_chat_action(context.chat_id, ChatAction.UPLOAD_PHOTO)

        file_path = f'{tmpdirname}/{file_id}.{"png" if is_png else "jpeg"}'
        im_file = bot.getFile(file_id)
        im_file.download(file_path)

        # Load image
        im_array = np.array(Image.open(file_path))
        im = Image.open(file_path)
        im.convert('RGB')
        pixels = im.load()
        width, height = im.size

        dict_size = {
            'width': width,
            'height': height
        }

        # Make sticker
        sticker = Image.new('RGBA', (width, height), (255, 0, 0, 0))
        sticker_pixels = sticker.load()

        # Segment image
        if segmentation:
            classes = [8, 12, 15]
            result = generate_labels(im_array)
            mask = np.isin(result, classes)
            result[mask] = 1

            for i in range(width):
                for j in range(height):
                    if result[j, i] != 0:
                        sticker_pixels[i, j] = pixels[i, j]

        else:
            sticker = im

        # Rescale sticker
        size_sorted = sorted(dict_size.items(), key=lambda kv: kv[1], reverse=True)
        max_ = size_sorted[0][1]
        min_ = size_sorted[1][1]

        rescale = {
            size_sorted[0][0]: 512,
            size_sorted[1][0]: int(min_ - min_ * (max_ - 512) / max_) if max_ < 512 else int(
                min_ + min_ * (512 - max_) / max_)
        }

        destination_path = f'{tmpdirname}/output.png'
        sticker.resize((rescale['width'], rescale['height'])).save(destination_path, 'PNG', optimize=True)

        add_sticker(bot, context, destination_path)
        # context.bot.send_document(chat_id=update.message.chat_id, document=open(destination_path, 'rb'))


def add_sticker(bot, context, file_path):
    with open(file_path, 'rb') as sticker:
        file = bot.upload_sticker_file(context.user_id, sticker)
        logger.info(f'https://t.me/addstickers/{context.sticker_set_name}')

        # La spéciale connie
        try:
            try:
                bot.add_sticker_to_set(context.user_id,
                                       context.sticker_set_name,
                                       file.file_id,
                                       context.emoji)
            except NetworkError:
                logging.info(f'New sticker pack created: {context.sticker_set_name}')
                sticker_set_title = f"{context.user_firstname}'s finest"
                bot.create_new_sticker_set(context.user_id,
                                           context.sticker_set_name,
                                           sticker_set_title,
                                           file.file_id,
                                           context.emoji)
        except (BadRequest, NetworkError):
            pass
        finally:
            try:
                sticker_set = bot.get_sticker_set(context.sticker_set_name)
                bot.send_sticker(context.chat_id, sticker_set.stickers[-1])
            except BadRequest:
                text = 'rekt.'
                logger.info(text)
                bot.send_message(context.chat_id, text=text)


def delete_sticker_set(bot, context):
    try:
        sticker_set = bot.get_sticker_set(context.sticker_set_name)
        size_sticker_set = len(sticker_set.stickers)
    except Exception:
        text = "You do not have a sticker pack yet. Please add a sticker first by sending me a picture."
        bot.send_message(context.chat_id, text=text)
    else:
        if size_sticker_set:
            for _ in range(size_sticker_set):
                bot.delete_sticker_from_set(bot.get_sticker_set(context.sticker_set_name).stickers[-1].file_id)
            text = "Your sticker pack was deleted!"
        else:
            text = "Cannot delete the sticker set, it is already empty."
        bot.send_message(context.chat_id, text=text)


def delete_sticker(bot, context, command):
    try:
        sticker_set = bot.get_sticker_set(context.sticker_set_name)
        size_sticker_set = len(sticker_set.stickers)
    except Exception:
        text = "You do not have a sticker pack yet. Please add a sticker first by sending me a picture."
    else:
        if not size_sticker_set:
            text = "Cannot delete the sticker, the set is already empty."
        else:
            logger.info(command)
            if not command.split()[0] == '/delete':
                text = "The command needs to be /delete."
            else:
                if len(command.split()) > 2:
                    text = "You passed too many arguments. Please send only one sticker index."
                elif len(command.split()) == 2:
                    if command.split()[1] == 'all':
                        for _ in range(size_sticker_set):
                            bot.delete_sticker_from_set(
                                bot.get_sticker_set(context.sticker_set_name).stickers[-1].file_id
                            )
                        text = "Your sticker pack was deleted!"
                    elif not command.split()[1].isdigit():
                        text = "The argument passed was not a number."
                    else:
                        index = int(command.split()[1])
                        if index > 0 and index <= size_sticker_set:
                            bot.delete_sticker_from_set(
                                bot.get_sticker_set(context.sticker_set_name).stickers[index - 1].file_id
                            )
                            text = "Your sticker was deleted!"
                        else:
                            text = f"Sticker index must be between 1 and {size_sticker_set}."

                elif len(command.split()) == 1:
                    bot.delete_sticker_from_set(bot.get_sticker_set(context.sticker_set_name).stickers[-1].file_id)
                    text = f"Your last sticker was deleted!"
    bot.send_message(context.chat_id, text=text)


def test(bot, update, context):
    bot.send_chat_action(context.chat_id, ChatAction.UPLOAD_PHOTO, timeout=50)


if __name__ == '__main__':
    app.run()
