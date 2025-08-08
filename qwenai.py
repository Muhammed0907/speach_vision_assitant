"""
Qwen AI Module - Dashscope Integration
Handles all Dashscope Qwen model interactions including:
- Emotion detection (qwen-vl-plus)
- Gender detection (qwen-vl-max)
- Text-to-Speech (cosyvoice-v1)
- LLM conversations (qwen-plus)
"""

import os
import sys
import threading
import time
import base64
import cv2
from dotenv import load_dotenv

# Dashscope imports
import dashscope
from dashscope.audio.tts_v2 import *
from dashscope import Generation
from openai import OpenAI

# Project imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './utils'))
from chat import CHAT_HISTORY, SYSTEM_PROMPT
from RealtimeMp3Player import RealtimeMp3Player
from echocheck import is_likely_system_echo

class QwenAI:
    def __init__(self):
        """Initialize Qwen AI with configuration from .env"""
        load_dotenv()
        
        # Load configuration
        self.model_use = os.getenv('MODEL_USE', 'qwen')
        self.api_key = os.getenv('MODEL_API_KEY')
        
        # Fallback to DASHSCOPEAPIKEY if MODEL_API_KEY is not found
        if not self.api_key:
            self.api_key = os.getenv('DASHSCOPEAPIKEY', 'sk-327233dd8f1f4012a7b25283b5da673d')
        
        # Initialize Dashscope
        self.init_dashscope_api_key()
        
        # Initialize OpenAI client for Dashscope endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # TTS/Speech related variables
        self.last_assistant_response = ""
        self.stop_event = threading.Event()
        self.now_speaking = threading.Lock()
        
        print(f"QwenAI initialized with model: {self.model_use}")
        print(f"Using API key: {self.api_key[:8]}...")
    
    def init_dashscope_api_key(self):
        """
        Set DashScope API-key from environment variables
        """
        if 'DASHSCOPEAPIKEY' in os.environ:
            dashscope.api_key = os.environ['DASHSCOPEAPIKEY']
        else:
            # Use MODEL_API_KEY if DASHSCOPEAPIKEY not found
            dashscope.api_key = self.api_key
            print(f"Using MODEL_API_KEY for Dashscope: {self.api_key[:8]}...")
    
    def encode_image(self, image_data):
        """Encode image data to base64 string"""
        _, buffer = cv2.imencode('.jpg', image_data)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_emotion_response(self, frame, products):
        """
        Get emotion-based response from qwen-vl-plus model
        
        Args:
            frame: OpenCV image frame
            products: List of available products
            
        Returns:
            str: Emotion-based product recommendation or None if error
        """
        try:
            print("SAY CHEESE: EMOTION CHECKING...")
            time.sleep(1)
            cv2.imwrite(f"./saved_frames/emotion_result_{time.time()}.jpg", frame)
            base64_image = self.encode_image(frame)
            
            # Create products string from array
            products_str = ",".join(products) if products else "饮品"
            print(f"PRODUCTS: {products_str}")
            
            prompt_text = f"""
            在图片上的客户的图片中，根据客户当前的情绪，从 {products_str} 中选择一款合适的饮品，用一句有共鸣、富有情感的句子进行推荐。不要总是使用"提提神"，而是根据不同情绪表达不同的语气和用词。还不用说'从图片来看'直接开始 您看起来
            """
            print(f"PROMPT IMAGE: {prompt_text}")
            
            completion = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{"role": "user","content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt_text},
                    ]}]
            )
            
            emotion_result = completion.choices[0].message.content
            return emotion_result
            
        except Exception as e:
            print(f"Emotion API error: {e}")
            return None
    
    def get_llm_gender(self, frame):
        """
        Get gender detection from qwen-vl-max model
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            str: 'M' for male, 'F' for female, 'unknown' for uncertain
        """
        try:
            base64_image = self.encode_image(frame)
            
            completion = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user","content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "这个人是男生还是女生？只回答：男 或者 女"},
                    ]}]
            )
            
            gender_result = completion.choices[0].message.content.strip()
            
            # Convert LLM response to expected format
            if "男" in gender_result:
                return 'M'
            elif "女" in gender_result:
                return 'F'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"LLM Gender API error: {e}")
            return 'unknown'
    
    def synthesis_text_to_speech_and_play_by_streaming_mode(self, text):
        """
        Synthesize speech with given text by streaming mode and play audio in real-time
        Uses cosyvoice-v1 model
        
        Args:
            text: Text to synthesize and play
        """
        # Update the last assistant response for echo detection
        self.last_assistant_response = text
        
        player = RealtimeMp3Player(verbose=True)
        # Start player with error handling
        try:
            player.start()
        except Exception as e:
            print(f"Failed to initialize audio player: {e}")
            return
        
        complete_event = threading.Event()
        
        # Capture the outer self reference for the callback
        outer_self = self
        
        class Callback(ResultCallback):
            def on_open(self):
                print('websocket is open.')
            
            def on_complete(self):
                print('speech synthesis task complete successfully.')
                complete_event.set()
            
            def on_error(self, message: str):
                print(f'speech synthesis task failed, {message}')
            
            def on_close(self):
                print('websocket is closed.')
            
            def on_event(self, message):
                pass
            
            def on_data(self, data: bytes) -> None:
                # Access stop_event from the captured outer self
                if not outer_self.stop_event.is_set():
                    player.write(data)
        
        # Initialize speech synthesizer
        synthesizer_callback = Callback()
        speech_synthesizer = SpeechSynthesizer(
            model='cosyvoice-v1',
            voice='longke',
            callback=synthesizer_callback
        )
        
        speech_synthesizer.call(text)
        print('Synthesized text: {}'.format(text))
        complete_event.wait()
        player.stop()
        print('[Metric] requestId: {}, first package delay ms: {}'.format(
            speech_synthesizer.get_last_request_id(),
            speech_synthesizer.get_first_package_delay()))
    
    class TTSCallback(ResultCallback):
        """Callback for TTS operations in LLM_Speak"""
        def __init__(self, qwen_ai_instance, player):
            self.qwen_ai = qwen_ai_instance
            self.player = player
        
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
            if not self.qwen_ai.stop_event.is_set():
                self.player.write(data)
    
    def llm_speak(self, system_prompt, user_query_queue):
        """
        LLM conversation handler using qwen-plus model
        Processes user queries and generates spoken responses
        
        Args:
            system_prompt: System prompt for the LLM
            user_query_queue: Queue containing user queries
        """
        global CHAT_HISTORY
        
        # Ensure system prompt is in history
        if not CHAT_HISTORY or CHAT_HISTORY[0]['role'] != 'system':
            CHAT_HISTORY.insert(0, {'role': 'system', 'content': system_prompt})
        
        while True:
            qr_txt = user_query_queue.get()  # Block until new message
            if qr_txt == "":
                continue
                
            print(f"qrTxt: {qr_txt}, LAST_ASSISTANT_RESPONSE: {self.last_assistant_response}")
            
            # Filter system echoes and short messages
            if is_likely_system_echo(qr_txt, self.last_assistant_response):
                print("Filtered: System echo")
                continue
            if len(qr_txt) < 4:
                print("Filtered: Too short")
                continue
            
            # Acquire speaking lock before processing
            self.now_speaking.acquire()
            try:
                # Reset and start new TTS session
                self.stop_event.clear()
                player = RealtimeMp3Player(verbose=True)
                try:
                    player.start()
                except Exception as e:
                    print(f"Failed to initialize audio player: {e}")
                    continue
                
                callback = self.TTSCallback(self, player)
                synthesizer = SpeechSynthesizer(
                    model='cosyvoice-v1',
                    voice='loongstella',
                    callback=callback
                )
                
                CHAT_HISTORY.append({'role': 'user', 'content': qr_txt})
                combined_text = ''
                
                # Generate response using qwen-plus
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
                    if self.stop_event.is_set():
                        synthesizer.streaming_complete()
                        break
                
                if 'NO_RESPONSE_NEEDED' in combined_text.upper():
                    CHAT_HISTORY.pop()
                    print("Filtered: NO_RESPONSE_NEEDED")
                    continue
                else:
                    CHAT_HISTORY.append({'role': 'assistant', 'content': combined_text})
                    self.last_assistant_response = combined_text
                
                synthesizer.streaming_complete()
            except Exception as e:
                print(f"Error in LLM_Speak: {e}")
            finally:
                try:
                    player.stop()
                except Exception as e:
                    print(f"Error in player.stop: {e}")
                # Ensure lock is released
                self.now_speaking.release()


# Global instance for backward compatibility
qwen_ai = QwenAI()

# Convenience functions for backward compatibility
def get_emotion_response(frame, products):
    """Convenience function for emotion detection"""
    return qwen_ai.get_emotion_response(frame, products)

def get_llm_gender(frame):
    """Convenience function for gender detection"""
    return qwen_ai.get_llm_gender(frame)

def synthesis_text_to_speech_and_play_by_streaming_mode(text):
    """Convenience function for TTS"""
    return qwen_ai.synthesis_text_to_speech_and_play_by_streaming_mode(text)

def init_dashscope_api_key():
    """Convenience function for API key initialization"""
    return qwen_ai.init_dashscope_api_key()

def LLM_Speak(system_prompt, user_query_queue=None):
    """Convenience function for LLM speak"""
    if user_query_queue is None:
        # Import from speak.py for backward compatibility
        from speak import userQueryQueue
        user_query_queue = userQueryQueue
    return qwen_ai.llm_speak(system_prompt, user_query_queue)


if __name__ == "__main__":
    # Test the module
    print("QwenAI module initialized successfully!")
    print(f"Model: {qwen_ai.model_use}")
    print(f"API Key: {qwen_ai.api_key[:8]}...")