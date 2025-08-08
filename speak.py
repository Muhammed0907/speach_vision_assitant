# !/usr/bin/env python3
# Copyright (C) Alibaba Group. All Rights Reserved.
# MIT License (https://opensource.org/licenses/MIT)

import os
import sys
import threading

import dashscope
from dashscope.audio.tts_v2 import *
from dashscope import Generation

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 './utils'))
from chat import CHAT_HISTORY,SYSTEM_PROMPT
from RealtimeMp3Player import RealtimeMp3Player

import multiprocessing
from echocheck import is_likely_system_echo
from dotenv import load_dotenv

from qwenai import init_dashscope_api_key, synthesis_text_to_speech_and_play_by_streaming_mode, LLM_Speak as qwen_llm_speak

userQueryQueue = multiprocessing.Queue()
LAST_ASSISTANT_RESPONSE = ""
STOP_EVENT = threading.Event()
NOW_SPEAKING = threading.Lock()
USER_ABSENT = threading.Event()  # Set when user is absent or too far away
SHOULD_LISTEN = threading.Event()  # Set when microphone should be listening (controlled by API)
SHOULD_LISTEN.set()  # Default to listening enabled

# Legacy wrapper for backward compatibility
def LLM_Speak(system_prompt):
    return qwen_llm_speak(system_prompt, userQueryQueue)

text_to_synthesize = '想不到时间过得这么快！昨天和你视频聊天，看到你那自豪又满意的笑容，我的心里呀，就如同喝了一瓶蜜一样甜呢！真心为你开心呢！'


# All TTS and LLM functionality has been moved to qwenai.py
# This file now serves as a compatibility layer

# All TTS and LLM functionality has been moved to qwenai.py
# Remaining legacy code commented out for reference

# main function
if __name__ == '__main__':
    init_dashscope_api_key()
    # while True:
    #     text_to_synthesize = input("Enter text to synthesize: ")
    #     synthesis_text_to_speech_and_play_by_streaming_mode(
    #         text=text_to_synthesize)

    try:
        LLM_Speak(SYSTEM_PROMPT)
    except KeyboardInterrupt:
        print("\nStopping on user interrupt…")
    finally:
        print("done")
