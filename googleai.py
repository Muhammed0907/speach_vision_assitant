"""
Google AI Module - Google Gemini Integration
Handles all Google AI interactions including:
- ASR (Google Cloud Speech-to-Text)
- TTS (Whisper-based synthesis)
- Gender detection (Gemini)
- LLM conversations (Gemini)
- Emotion detection (Gemini)
"""

import os
import sys
import threading
import time
import base64
import cv2
import hashlib
import json
import io
import wave
from pathlib import Path
from dotenv import load_dotenv
import pygame
import pyaudio
import numpy as np
import queue

# Google imports
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

# Project imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './utils'))
from chat import CHAT_HISTORY, SYSTEM_PROMPT
from echocheck import is_likely_system_echo

class GoogleAI:
    def __init__(self):
        """Initialize Google AI with configuration from .env"""
        load_dotenv()
        
        # Load configuration
        self.model_use = os.getenv('MODEL_USE', 'google')
        self.api_key = os.getenv('MODEL_API_KEY', 'AIzaSyBoAEFmLXyF75KhrXO9twwBjCguArL6JQs')
        
        # Initialize Google Generative AI client
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize Whisper for ASR
        self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        # ASR related variables
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Audio parameters for ASR
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 3  # Process every 3 seconds
        
        # TTS/Speech related variables
        self.last_assistant_response = ""
        self.stop_event = threading.Event()
        self.now_speaking = threading.Lock()
        
        # TTS Cache setup
        self.cache_dir = Path("tts_cache_google")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
        except pygame.error as e:
            print(f"Warning: Could not initialize pygame mixer: {e}")
        
        print(f"GoogleAI initialized with model: {self.model_use}")
        print(f"Using API key: {self.api_key[:8]}...")
        print(f"TTS cache initialized at: {self.cache_dir}")
        print(f"Cache contains {len(self.cache_index)} entries")
        print(f"Whisper ASR model loaded: tiny")
    
    def _load_cache_index(self):
        """Load the TTS cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache index: {e}")
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save the TTS cache index to disk"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {e}")
    
    def _get_text_hash(self, text, voice='Kore', model='gemini-tts'):
        """Generate a hash for the text and voice combination"""
        hash_input = f"{text}_{voice}_{model}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def _is_cache_valid(self, cache_path):
        """Check if a cached file exists and is valid"""
        return cache_path.exists() and cache_path.stat().st_size > 0
    
    def _get_cached_audio(self, text_hash):
        """Get cached audio file path if it exists and is valid"""
        if text_hash in self.cache_index:
            cache_info = self.cache_index[text_hash]
            cache_path = Path(cache_info['file_path'])
            if self._is_cache_valid(cache_path):
                return cache_path
            else:
                # Remove invalid cache entry
                del self.cache_index[text_hash]
                self._save_cache_index()
        return None
    
    def _save_to_cache(self, text_hash, text, audio_data, voice='Kore', model='gemini-tts'):
        """Save audio data to cache"""
        try:
            cache_filename = f"{text_hash}.wav"
            cache_path = self.cache_dir / cache_filename
            
            with open(cache_path, 'wb') as f:
                f.write(audio_data)
            
            # Update cache index
            self.cache_index[text_hash] = {
                'text': text,
                'voice': voice,
                'model': model,
                'file_path': str(cache_path),
                'created_at': time.time()
            }
            self._save_cache_index()
            
            print(f"Cached TTS for text: '{text[:50]}...' -> {cache_filename}")
            return cache_path
        except Exception as e:
            print(f"Error saving to cache: {e}")
            return None
    
    def _play_cached_audio(self, cache_path):
        """Play cached audio file using pygame"""
        try:
            pygame.mixer.music.load(str(cache_path))
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                pygame.time.wait(100)
            
            return True
        except Exception as e:
            print(f"Error playing cached audio: {e}")
            return False
    
    def _wave_file_from_data(self, pcm_data, channels=1, rate=24000, sample_width=2):
        """Create wave file data from PCM data"""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
        return buffer.getvalue()

    def encode_image(self, image_data):
        """Encode image data to base64 string"""
        _, buffer = cv2.imencode('.jpg', image_data)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_emotion_response(self, frame, products):
        """
        Get emotion-based response from Gemini model
        
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
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"text": prompt_text},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }}
                ]
            )
            
            emotion_result = response.text
            return emotion_result
            
        except Exception as e:
            print(f"Emotion API error: {e}")
            return None
    
    def get_llm_gender(self, frame):
        """
        Get gender detection from Gemini model
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            str: 'M' for male, 'F' for female, 'unknown' for uncertain
        """
        try:
            base64_image = self.encode_image(frame)
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"text": "这个人是男生还是女生？只回答：男 或者 女"},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }}
                ]
            )
            
            gender_result = response.text.strip()
            
            # Convert response to expected format
            if "男" in gender_result:
                return 'M'
            elif "女" in gender_result:
                return 'F'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"LLM Gender API error: {e}")
            return 'unknown'
    
    def synthesis_text_to_speech_and_play_by_streaming_mode(self, text, voice='Kore', model='gemini-tts'):
        """
        Synthesize speech with given text using Google's TTS and play audio in real-time
        Uses Google Gemini TTS with caching support
        
        Args:
            text: Text to synthesize and play
            voice: Voice to use (default: 'Kore')
            model: TTS model to use (default: 'gemini-tts')
        """
        # Update the last assistant response for echo detection
        self.last_assistant_response = text
        
        # Check cache first
        text_hash = self._get_text_hash(text, voice, model)
        cached_audio_path = self._get_cached_audio(text_hash)
        
        if cached_audio_path:
            print(f'Playing cached TTS for: "{text[:50]}..."')
            if self._play_cached_audio(cached_audio_path):
                return
            else:
                print("Failed to play cached audio, generating new TTS...")
        
        print(f'Generating TTS for: "{text[:50]}..."')
        
        try:
            # Generate audio using Google Gemini TTS
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=f"Say cheerfully: {text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                )
            )
            
            # Get audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Create wave file data
            wave_data = self._wave_file_from_data(audio_data)
            
            # Save to cache
            cache_path = self._save_to_cache(text_hash, text, wave_data, voice, model)
            
            # Play audio using pygame
            buffer = io.BytesIO(wave_data)
            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                pygame.time.wait(100)
            
        except Exception as e:
            print(f"Error in TTS synthesis: {e}")
    
    def llm_speak(self, system_prompt, user_query_queue):
        """
        LLM conversation handler using Gemini model
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
                
                CHAT_HISTORY.append({'role': 'user', 'content': qr_txt})
                
                # Generate response using Gemini
                try:
                    # Convert chat history to Gemini format
                    messages_content = []
                    for msg in CHAT_HISTORY:
                        if msg['role'] == 'system':
                            messages_content.append(f"System: {msg['content']}")
                        elif msg['role'] == 'user':
                            messages_content.append(f"User: {msg['content']}")
                        elif msg['role'] == 'assistant':
                            messages_content.append(f"Assistant: {msg['content']}")
                    
                    full_prompt = "\n".join(messages_content)
                    
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=full_prompt
                    )
                    
                    combined_text = response.text
                    
                    if 'NO_RESPONSE_NEEDED' in combined_text.upper():
                        CHAT_HISTORY.pop()
                        print("Filtered: NO_RESPONSE_NEEDED")
                        continue
                    elif combined_text.strip() == 'N':
                        CHAT_HISTORY.pop()
                        print("Filtered: Single N response")
                        continue
                    else:
                        CHAT_HISTORY.append({'role': 'assistant', 'content': combined_text})
                        self.last_assistant_response = combined_text
                        
                        # Generate and play TTS
                        self.synthesis_text_to_speech_and_play_by_streaming_mode(combined_text)
                    
                except Exception as e:
                    print(f"Error in Gemini response generation: {e}")
                    CHAT_HISTORY.pop()  # Remove the user message if response failed
                
            except Exception as e:
                print(f"Error in LLM_Speak: {e}")
            finally:
                # Ensure lock is released
                self.now_speaking.release()
    
    def get_cache_stats(self):
        """Get TTS cache statistics"""
        total_entries = len(self.cache_index)
        total_size = 0
        valid_entries = 0
        
        for text_hash, cache_info in self.cache_index.items():
            cache_path = Path(cache_info['file_path'])
            if self._is_cache_valid(cache_path):
                valid_entries += 1
                total_size += cache_path.stat().st_size
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'invalid_entries': total_entries - valid_entries,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir)
        }
    
    def clear_cache(self, older_than_days=None):
        """Clear TTS cache entries
        
        Args:
            older_than_days: If specified, only clear entries older than this many days
        """
        cleared_count = 0
        current_time = time.time()
        
        entries_to_remove = []
        
        for text_hash, cache_info in self.cache_index.items():
            should_remove = False
            
            if older_than_days:
                created_at = cache_info.get('created_at', 0)
                age_days = (current_time - created_at) / (24 * 3600)
                if age_days > older_than_days:
                    should_remove = True
            else:
                should_remove = True
            
            if should_remove:
                # Remove file
                cache_path = Path(cache_info['file_path'])
                if cache_path.exists():
                    cache_path.unlink()
                entries_to_remove.append(text_hash)
                cleared_count += 1
        
        # Update index
        for text_hash in entries_to_remove:
            del self.cache_index[text_hash]
        
        self._save_cache_index()
        print(f"Cleared {cleared_count} cache entries")
        return cleared_count
    
    def start_asr_recording(self, user_query_queue):
        """Start ASR recording using Whisper model - only when user is present"""
        self.is_recording = True
        audio = pyaudio.PyAudio()
        
        print(f"Available audio devices: {audio.get_device_count()}")
        
        # Import user presence controls
        try:
            from speak import SHOULD_LISTEN, USER_ABSENT
        except ImportError:
            print("Warning: Could not import user presence controls")
            # Create dummy events if import fails
            import threading
            SHOULD_LISTEN = threading.Event()
            USER_ABSENT = threading.Event()
            SHOULD_LISTEN.set()
        
        try:
            stream = audio.open(format=self.format,
                               channels=self.channels,
                               rate=self.rate,
                               input=True,
                               frames_per_buffer=self.chunk)
            
            print("Google AI ASR Recording... (Whisper)")
            print("Audio stream opened successfully")
            print("Microphone will only listen when user is present")
            
            while self.is_recording:
                # Check if we should listen (API control and user presence)
                if not SHOULD_LISTEN.is_set():
                    print("Microphone disabled by API - waiting...")
                    time.sleep(1)
                    continue
                
                if USER_ABSENT.is_set():
                    print("No user detected - microphone paused")
                    time.sleep(1)
                    continue
                
                print("User present - microphone active")
                frames = []
                for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                    if not self.is_recording:
                        break
                    # Double-check user presence during recording
                    if USER_ABSENT.is_set() or not SHOULD_LISTEN.is_set():
                        print("User left or listening disabled during recording - stopping")
                        break
                    data = stream.read(self.chunk)
                    frames.append(data)
                
                if frames and self.is_recording and not USER_ABSENT.is_set() and SHOULD_LISTEN.is_set():
                    audio_data = b''.join(frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Check if audio has sufficient volume
                    audio_level = np.sqrt(np.mean(audio_np ** 2))
                    
                    if audio_level > 0.001:  # Minimum threshold
                        # Process with Whisper
                        try:
                            segments, info = self.whisper_model.transcribe(audio_np, beam_size=1)
                            
                            for segment in segments:
                                if segment.text.strip():
                                    transcribed_text = segment.text.strip()
                                    print(f"[ASR] User present - processing: {transcribed_text}")
                                    # Put transcribed text into the user query queue
                                    user_query_queue.put(transcribed_text)
                        except Exception as e:
                            print(f"Whisper transcription error: {e}")
                    else:
                        print("Audio level too low - skipping")
                else:
                    print("Skipping audio processing - user absent or listening disabled")
                    
        except Exception as e:
            print(f"ASR recording error: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
            self.is_recording = False
    
    def stop_asr_recording(self):
        """Stop ASR recording"""
        self.is_recording = False


# Global instance for backward compatibility
google_ai = GoogleAI()

# Convenience functions for backward compatibility
def get_emotion_response(frame, products):
    """Convenience function for emotion detection"""
    return google_ai.get_emotion_response(frame, products)

def get_llm_gender(frame):
    """Convenience function for gender detection"""
    return google_ai.get_llm_gender(frame)

def synthesis_text_to_speech_and_play_by_streaming_mode(text, voice='Kore', model='gemini-tts'):
    """Convenience function for TTS with caching support"""
    return google_ai.synthesis_text_to_speech_and_play_by_streaming_mode(text, voice, model)

def init_dashscope_api_key():
    """Convenience function for API key initialization (Google AI doesn't need this)"""
    print("Google AI initialized, no additional API key setup needed")
    return True

def LLM_Speak(system_prompt, user_query_queue=None):
    """Convenience function for LLM speak"""
    if user_query_queue is None:
        # Import from speak.py for backward compatibility
        try:
            from speak import userQueryQueue
            user_query_queue = userQueryQueue
        except ImportError:
            print("Warning: Could not import userQueryQueue from speak.py")
            import queue
            user_query_queue = queue.Queue()
    return google_ai.llm_speak(system_prompt, user_query_queue)

def get_tts_cache_stats():
    """Get TTS cache statistics"""
    return google_ai.get_cache_stats()

def clear_tts_cache(older_than_days=None):
    """Clear TTS cache"""
    return google_ai.clear_cache(older_than_days)

def mic_listen(user_query_queue=None):
    """Convenience function for ASR using Whisper"""
    if user_query_queue is None:
        try:
            from speak import userQueryQueue
            user_query_queue = userQueryQueue
        except ImportError:
            print("Warning: Could not import userQueryQueue from speak.py")
            import queue
            user_query_queue = queue.Queue()
    return google_ai.start_asr_recording(user_query_queue)

def stop_mic_listen():
    """Stop microphone listening"""
    return google_ai.stop_asr_recording()


if __name__ == "__main__":
    # Test the module
    print("GoogleAI module initialized successfully!")
    print(f"Model: {google_ai.model_use}")
    print(f"API Key: {google_ai.api_key[:8]}...")