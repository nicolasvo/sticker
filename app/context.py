import hashlib


class Context:

    def __init__(self, bot, update):
        self.chat_id = update.message.chat_id
        self.bot_username = bot.username
        self.user_id = update.message.from_user.id
        self.user_firstname = update.message.from_user.first_name
        hash = hashlib.md5(bytearray(self.user_id)).hexdigest()
        self.sticker_set_name = f'Stickermacher_{hash[:10]}_by_{self.bot_username}'
        self.emoji = '🤖'
