from functools import wraps
from telegram import ChatAction
import time
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def timeit(func):
    @wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        logger.info('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))

    return newfunc


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(update, context, *args, **kwargs)

        return command_func

    return decorator


send_typing_action = send_action(ChatAction.TYPING)
send_upload_photo_action = send_action(ChatAction.UPLOAD_PHOTO)
emojis = ['🦘', '🦆', '🦅', '🦉', '🦜', '🦄', '🐝', '🐢', '🐍', '🦖', '🦕', '🦀', '🦈', '🐊', '🐅', '🐆', '🦓', '🦍',
          '🐘', '🦏', '🦛', '🐪', '🦒', '🐃', '🐂', '🦌', '🐕', '🐈', '🐓', '🕊']
