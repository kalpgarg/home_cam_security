"""
 *  @file  telegram_bot.py
 *  @brief This file contains implemenation of sending message to telegram bot
 *
 *  @author Kalp Garg.
"""
#!/usr/bin/env python3

import requests
import argparse

import os
from common_utils import get_tgram_keys, return_datetime

class TBot():
    def __init__(self, cred_loc, chat='home_recordings'):

        if chat == "home_recordings":
            bot_id, chat_id, = get_tgram_keys(cred_loc)
        else:
            print("CHAT NAME DOESNT EXIST....")

        self.bot_token = bot_id
        self.chat_ID = chat_id

    def send_message(self, message):
        print(f"Message to be sent : {message}")
        send_text = 'https://api.telegram.org/bot' + self.bot_token + '/sendMessage?chat_id=' + self.chat_ID + '&parse_mode=HTML&text=' + message
        response = requests.get(send_text)
        print(f"Response received : {response.json()}")
        return response.json()
    
    def send_video(self, video_f_path, caption="video"):
        files = {'video': open(f'{video_f_path}', 'rb')}
        response = requests.post('https://api.telegram.org/bot' + self.bot_token + '/sendVideo?chat_id=' + self.chat_ID + '&caption=' + caption, files=files)
        sent = response.json()['ok']
        print(f"Response received : {sent}")
        return sent

if __name__ == "__main__":

    t_bot_args = argparse.ArgumentParser(description="Telegram bot")
    t_bot_args.version = "23.08.01"  # yy.mm.vv
    t_bot_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    t_bot_args.add_argument('-l', '--log_folder', type=str, metavar='telegram_bot_log',
                                   default="telegram_bot_log",
                                   help="Location of the log folder")
    t_bot_args.add_argument('-cl', '--cred_loc', action='store', metavar='cam_info.json', type=str,
                                 help='path of cam info file', required=True)    
    t_bot_args.add_argument('-msg', '--chat_msg', action='store', type= str, help='Telegram chat message', required=True)

    args = t_bot_args.parse_args()

    # tbot = TBot(bot_token= args.bot_token, chat_ID=args.chat_ID)
    # main(BOT_TOKEN)
    tbot = TBot(args.cred_loc, chat="home_recordings")
    tbot.send_video(video_f_path=args.chat_msg)
    




