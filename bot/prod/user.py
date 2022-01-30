import os
import hashlib


NUMERO_GAGNANT = os.getenv("NUMERO_GAGNANT")


class User:
    def __init__(self, update, bot):
        self.chat_id = update.message.chat_id
        self.bot_username = bot.username
        self.id = update.message.from_user.id
        self.firstname = update.message.from_user.first_name
        self.hash = hashlib.md5(bytearray(self.id+int(LOTO))).hexdigest()
        self.sticker_set_name = self.get_sticker_set_name(bot, 0)
        self.emoji = "💊"

    def get_sticker_set_name(self, bot, pack_number=1):
        sticker_set_name = f"Z_{pack_number}_{self.hash[:10]}_by_{self.bot_username}"
        if pack_number > 1:
            sticker_set_name = (
                f"Z_{pack_number}_{self.hash[:10]}_by_{self.bot_username}"
            )
            return sticker_set_name
        try:
            sticker_set = bot.get_sticker_set(sticker_set_name)
            if sticker_set:
                if len(sticker_set.stickers) == 120:
                    pack_number += 1
                    return self.get_sticker_set_name(bot, pack_number)
        except:
            pass

        return sticker_set_name
