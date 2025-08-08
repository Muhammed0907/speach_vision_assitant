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

userQueryQueue = multiprocessing.Queue()
LAST_ASSISTANT_RESPONSE = ""
STOP_EVENT = threading.Event()
NOW_SPEAKING = threading.Lock()
USER_ABSENT = threading.Event()  # Set when user is absent or too far away
SHOULD_LISTEN = threading.Event()  # Set when microphone should be listening (controlled by API)
SHOULD_LISTEN.set()  # Default to listening enabled

text_to_synthesize = '想不到时间过得这么快！昨天和你视频聊天，看到你那自豪又满意的笑容，我的心里呀，就如同喝了一瓶蜜一样甜呢！真心为你开心呢！'


def init_dashscope_api_key():
    '''
    Set your DashScope API-key. More information:
    https://github.com/aliyun/alibabacloud-bailian-speech-demo/blob/master/PREREQUISITES.md
    '''
    # Import dotenv to load environment variables from .env file
    
    # Load environment variables from .env file
    load_dotenv()
    
    # print("Environment variables:", os.environ)
    if 'DASHSCOPEAPIKEY' in os.environ:
        dashscope.api_key = os.environ[
            'DASHSCOPEAPIKEY']  # load API-key from environment variable DASHSCOPE_API_KEY
        # print(f"dashscope.api_key: {dashscope.api_key}")
    else:
        print("Warning: DASHSCOPE_API_KEY not found in environment variables")
        dashscope.api_key = '<your-dashscope-api-key>'  # set API-key manually

def synthesis_text_to_speech_and_play_by_streaming_mode(text):
    '''
    Synthesize speech with given text by streaming mode, async call and play the synthesized audio in real-time.
    for more information, please refer to https://help.aliyun.com/document_detail/2712523.html
    '''
    global LAST_ASSISTANT_RESPONSE
    # Update the last assistant response for echo detection
    LAST_ASSISTANT_RESPONSE = text
    
    player = RealtimeMp3Player(verbose=True)
    # start player with error handling
    try:
        player.start()
    except Exception as e:
        print(f"Failed to initialize audio player: {e}")
        return  # Skip TTS if audio fails

    complete_event = threading.Event()

    # Define a callback to handle the result

    class Callback(ResultCallback):
        def on_open(self):
            # self.file = open('result.mp3', 'wb')
            print('websocket is open.')

        def on_complete(self):
            print('speech synthesis task complete successfully.')
            complete_event.set()

        def on_error(self, message: str):
            print(f'speech synthesis task failed, {message}')

        def on_close(self):
            print('websocket is closed.')

        def on_event(self, message):
            # print(f'recv speech synthsis message {message}')
            pass

        def on_data(self, data: bytes) -> None:
            if not STOP_EVENT.is_set():
                player.write(data)
            # send to player
            # player.write(data)
            # save audio to file
            # self.file.write(data)

    # Call the speech synthesizer callback
    synthesizer_callback = Callback()

    # Initialize the speech synthesizer
    # you can customize the synthesis parameters, like voice, format, sample_rate or other parameters
    speech_synthesizer = SpeechSynthesizer(model='cosyvoice-v1',
                                           voice='longke',
                                           callback=synthesizer_callback)

    speech_synthesizer.call(text)
    print('Synthesized text: {}'.format(text))
    complete_event.wait()
    player.stop()
    print('[Metric] requestId: {}, first package delay ms: {}'.format(
        speech_synthesizer.get_last_request_id(),
        speech_synthesizer.get_first_package_delay()))

# def LLM_Speach(qrTxt:str,systemPrompt:str):
#     global CHAT_HISTORY
#     player = RealtimeMp3Player()
#     player.start()
#     class CallBack(ResultCallback):
#         def on_open(self) -> None:
#             pass

#         def on_complete(self) -> None:
#             pass

#         def on_error(self, message) -> None:
#             print(f'speech synthesis task failed, {message}')
        
#         def on_close(self) -> None:
#             pass

#         def on_event(self,message):
#             pass

#         def on_data(self,data:bytes) ->None:
#             player.write(data)
    
#     synthesizerCallBack = CallBack()


#     synthesizer = SpeechSynthesizer(
#         model='cosyvoice-v1',
#         voice='loongstella',
#         callback=synthesizerCallBack
#     )

#     if len(CHAT_HISTORY) == 0 or CHAT_HISTORY[0]['role'] != 'system':
#         CHAT_HISTORY.insert(0,{
#             'role': 'system',
#             'content': systemPrompt
#         })

#     CHAT_HISTORY.append({
#         'role': 'user',
#         'content': qrTxt
#     })
#     print(f"CHAT_HIST: {CHAT_HISTORY}\n")
#     # try:    
#     response = Generation.call(
#         model='qwen-plus',
#         messages=CHAT_HISTORY,
#         result_format='message',  # set result format as 'message'
#         stream=True,  # enable stream output
#         incremental_output=True,  # enable incremental output
#     )


#     llmResponsecontent = ""
#     for response in response:
#         if response.status_code == 200:
#             llmTextChunk =response.output.choices[0].message.content
#             if llmTextChunk == "N":
#                 return None
#             synthesizer.streaming_call(llmTextChunk)
#             print(llmTextChunk,end="",flush=True)
#             llmResponsecontent += llmTextChunk
            
#     if 'NO_RESPONSE_NEEDED' in llmResponsecontent.upper():
#         print("\n>>> Detected NO_RESPONSE_NEEDED in streaming response, stopping playback")
#         return None
    

#     # synthesizer.streaming_call(llmResponsecontent)
#     synthesizer.streaming_complete()
#     player.stop()

#     CHAT_HISTORY.append({
#         'role': 'assistant',
#         'content': llmResponsecontent
#     })

#     return True

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



# def LLM_Speak(systemPrompt: str):
#     global CHAT_HISTORY, LAST_ASSISTANT_RESPONSE, STOP_EVENT, synthesizer, player

#     # ensure system prompt is in history
#     if not CHAT_HISTORY or CHAT_HISTORY[0]['role'] != 'system':
#         CHAT_HISTORY.insert(0, {'role': 'system', 'content': systemPrompt})

#     while True:
#         qrTxt = userQueryQueue.get()   # BLOCK until new message
#         # if not NOW_SPEAKING.acquire(blocking=False):
#         #     print("Skipping response due to ongoing speech")
#         #     continue
#         if qrTxt == "":
#             continue
        
            
#         # reset and start new TTS session
#         STOP_EVENT.clear()
#         player = RealtimeMp3Player();  player.start()
#         callback = TTSCallback()
#         synthesizer = SpeechSynthesizer(model='cosyvoice-v1', voice='loongstella', callback=callback)

#         CHAT_HISTORY.append({'role': 'user', 'content': qrTxt})
#         combined_text = ''

#         # try:
#         for resp in Generation.call(
#                 model='qwen-plus',
#                 messages=CHAT_HISTORY,
#                 result_format='message',
#                 stream=True,
#                 incremental_output=True
#             ):
#             if resp.status_code != 200:
#                 continue
#             chunk = resp.output.choices[0].message.content
#             if chunk == 'N':   # your "no-response" sentinel
#                 CHAT_HISTORY.pop()

#                 # return
#                 break

#             synthesizer.streaming_call(chunk)
#             combined_text += chunk
#             if STOP_EVENT.is_set():
#                 synthesizer.streaming_complete()
#                 break

#         if 'NO_RESPONSE_NEEDED' in combined_text.upper():
#             CHAT_HISTORY.pop()
#             print("Filtered: NO_RESPONSE_NEEDED")
#             continue
#         else:
#             CHAT_HISTORY.append({'role': 'assistant', 'content': combined_text})
#             LAST_ASSISTANT_RESPONSE = combined_text
#         try:
#             synthesizer.streaming_complete()
#         except: pass
#         # try:
#         player.stop()


def LLM_Speak(systemPrompt: str):
    global CHAT_HISTORY, LAST_ASSISTANT_RESPONSE, STOP_EVENT, synthesizer, player

    # Ensure system prompt is in history
    if not CHAT_HISTORY or CHAT_HISTORY[0]['role'] != 'system':
        CHAT_HISTORY.insert(0, {'role': 'system', 'content': systemPrompt})

    while True:
        qrTxt = userQueryQueue.get()  # Block until new message
        if qrTxt == "":
            continue
        print(f"qrTxt: {qrTxt}, LAST_ASSISTANT_RESPONSE: {LAST_ASSISTANT_RESPONSE}")
        if is_likely_system_echo(qrTxt, LAST_ASSISTANT_RESPONSE):
            print("Filtered: System echo")
            continue
        if len(qrTxt) < 4:
            print("Filtered: Too short")
            continue
        # Acquire speaking lock before processing
        NOW_SPEAKING.acquire()
        try:
            # Reset and start new TTS session
            STOP_EVENT.clear()
            player = RealtimeMp3Player(verbose=True)
            try:
                player.start()
            except Exception as e:
                print(f"Failed to initialize audio player: {e}")
                continue  # Skip this response if audio fails
            callback = TTSCallback()
            synthesizer = SpeechSynthesizer(model='cosyvoice-v1', voice='loongstella', callback=callback)

            CHAT_HISTORY.append({'role': 'user', 'content': qrTxt})
            combined_text = ''

            # Generate response
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
                if chunk == 'N':
                    CHAT_HISTORY.pop()
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

            synthesizer.streaming_complete()
        except Exception as e:
            print(f"Error in LLM_Speak: {e}")
        finally:
            try:
                player.stop()
            except Exception as e:
                print(f"Error in player.stop: {e}")
                pass
            # Ensure resources are cleaned up and lock is released
            NOW_SPEAKING.release()

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
