import os
import hashlib

NUMERO_GAGNANT = os.getenv("NUMERO_GAGNANT")


class User:
    def __init__(self, update, bot):
        self.chat_id = update.message.chat_id
        self.bot_username = bot.username
        self.id = update.message.from_user.id
        self.firstname = update.message.from_user.first_name
        # TODO: remove hack
        if len(str(self.id)) > 9:
            self.hash = hashlib.md5(bytearray((str(self.id) + NUMERO_GAGNANT).encode("utf-8"))).hexdigest()
        else:
            self.hash = hashlib.md5(bytearray(self.id + int(NUMERO_GAGNANT))).hexdigest()
        self.sticker_set_name = self.get_sticker_set_name(bot, 0)
        self.sticker_set_title = self.get_sticker_set_title()
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

    def get_sticker_set_title(self):
        def sarcastic(string, start_lower=True):
            if start_lower:
                res = [
                    l.upper() if index % 2 else l
                    for index, l in enumerate(string.lower())
                ]
            else:
                res = [
                    l if index % 2 else l.upper()
                    for index, l in enumerate(string.lower())
                ]
            return "".join(res)

        if len(self.firstname) % 2:
            sticker_set_title = "'S fInEsT"
        else:
            sticker_set_title = "'s FiNeSt"

        return sarcastic(self.firstname) + sticker_set_title
