import sys
import dashscope
from dashscope.audio.tts_v2 import *
import os
import threading
from dashscope import Generation

sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./utils"
))
from chat import SYSTEM_PROMPT
from RealtimeMp3Player import RealtimeMp3Player
from echocheck import is_likely_system_echo
import multiprocessing
qrQueue = multiprocessing.Queue()
# SYSTEM_PROMPT =  """你是一个友好热情的盲合活动组织者。你的主要任务是欢迎参与者，推荐盲合活动形式，并与他们进行愉快的聊天。

# 重要过滤指令：

# 如果你认为用户的问题不是直接对盲合活动组织者说的，回复 “NO_RESPONSE_NEEDED”

# 如果用户的问题与盲合、活动、配对或社交无关，回复 “NO_RESPONSE_NEEDED”

# 如果问题听起来像是参与者之间的对话而非对你说的，回复 “NO_RESPONSE_NEEDED”

# 如果听起来是随意的闲聊或背景对话，回复 “NO_RESPONSE_NEEDED”

# 只有当你确定用户是在对盲合活动组织者说话时，才提供正常回复。
# 当你正常回复时，使用热情活泼的语气，但保持简短的回答。你可以推荐的盲合活动包括：快速闪配、主题小组、视频速配和户外漫步配对。
# 记住，当你决定回复时，你的目标是让参与者感到受欢迎并愿意参加一次盲合。每次回复都应该友好且积极，不要使用过于正式的语言。"""

CHAT_HISTORY = []
LAST_ASSISTANT_RESPONSE = ""
# Use an Event for thread-safe signaling
STOP_EVENT = threading.Event()
synthesizer = None
player = None

class TTSCallback(ResultCallback):
    def on_open(self):
        pass

    def on_complete(self):
        print("speech synthesis task completed")

    def on_error(self, message):
        print(f'speech synthesis task failed: {message}')

    def on_close(self):
        print("speech synthesis task closed")

    def on_event(self, message):
        pass

    def on_data(self, data: bytes):
        # Write audio only if not stopped
        if not STOP_EVENT.is_set():
            player.write(data)



def LLM_Speak(systemPrompt: str):
    global CHAT_HISTORY, LAST_ASSISTANT_RESPONSE, STOP_EVENT, synthesizer, player

    # ensure system prompt is in history
    if not CHAT_HISTORY or CHAT_HISTORY[0]['role'] != 'system':
        CHAT_HISTORY.insert(0, {'role': 'system', 'content': systemPrompt})

    while True:
        qrTxt = qrQueue.get()   # BLOCK until new message
        if qrTxt == "":
            continue
        

        # reset and start new TTS session
        STOP_EVENT.clear()
        player = RealtimeMp3Player();  player.start()
        callback = TTSCallback()
        synthesizer = SpeechSynthesizer(model='cosyvoice-v1', voice='loongstella', callback=callback)

        CHAT_HISTORY.append({'role': 'user', 'content': qrTxt})
        combined_text = ''

        # try:
        for resp in Generation.call(
                model='qwen-plus',
                messages=CHAT_HISTORY,
                result_format='message',
                stream=True,
                incremental_output=True
            ):
            if resp.status_code != 200:
                continue
            chunk = resp.output.choices[0].message.content
            if chunk == 'N':   # your “no-response” sentinel
                CHAT_HISTORY.pop()

                # return
                break

            synthesizer.streaming_call(chunk)
            combined_text += chunk
            if STOP_EVENT.is_set():
                synthesizer.streaming_complete()
                break

        if 'NO_RESPONSE_NEEDED' in combined_text.upper():
            CHAT_HISTORY.pop()
            print("Filtered: NO_RESPONSE_NEEDED")
            continue
        else:
            CHAT_HISTORY.append({'role': 'assistant', 'content': combined_text})
            LAST_ASSISTANT_RESPONSE = combined_text
        try:
            synthesizer.streaming_complete()
        except: pass
        # try:
        player.stop()


def main():
    print("输入 's' 停止当前合成并播放。按任意其他键开始新的合成。")
    threading.Thread(target=LLM_Speak, args=(SYSTEM_PROMPT,), daemon=True).start()

    while True:
        try:
            user_input = input("Enter text to synthesize: ")
            if is_likely_system_echo(user_input, LAST_ASSISTANT_RESPONSE):
                print("Detected system echo, skipping response")
                # continue
            else:   
                STOP_EVENT.set()          # signal “stop current speech”
                # STOP_EVENT.clear()
                qrQueue.put(user_input)   # hand off to the single LLM_Speak loop
        except KeyboardInterrupt:
            print("Stopping on user interrupt…")
            break


if __name__ == '__main__':
    # 
    print(SYSTEM_PROMPT)
    from fetchDataFromAPI import fetch_product_by_name
    result = fetch_product_by_name("MangHe")
    if result.get("prompt"):
        SYSTEM_PROMPT = result.get("prompt")
    print(SYSTEM_PROMPT)
    main()
    # print(result)
