from fetchDataFromAPI import fetch_product_by_name, check_listen_status
import sys
from listener import mic_listen
import cv2
import time
import threading
import argparse
import numpy as np
import queue
from collections import deque
from insightface.app import FaceAnalysis
from task_monitor import TaskMonitor
import os
import glob
from datetime import datetime, timedelta
import base64
from openai import OpenAI
import subprocess

from speak import ( init_dashscope_api_key, 
                    synthesis_text_to_speech_and_play_by_streaming_mode, 
                    LLM_Speak, 
                    userQueryQueue, 
                    LAST_ASSISTANT_RESPONSE, 
                    STOP_EVENT, 
                    NOW_SPEAKING,
                    USER_ABSENT,
                    SHOULD_LISTEN)

from greetings import (male_greetings, 
                       female_greetings, 
                       neutral_greetings)
from suggestion import AUTO_SUGGESTIONS
from chat import SYSTEM_PROMPT, NO_RESPONSE_NEEDED_RULE, default
from echocheck import is_likely_system_echo
import random
import os
import multiprocessing

# Seed random number generator for better randomness
random.seed()

def increase_audio_volume(volume_percent=100):
    """Set system audio volume using amixer for all audio cards"""
    try:
        # Validate volume range
        volume_percent = max(0, min(100, volume_percent))
        
        # Get available mixer controls
        result = subprocess.run(['amixer', 'scontrols'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            controls = result.stdout.strip().split('\n')
            print(f"Available audio controls: {controls}")
            
            # Set volume for all possible audio cards and controls
            audio_cards = ['0', '1', '2', '3']  # Based on your aplay -l output
            control_names = ['Master', 'PCM', 'Speaker', 'Headphone', 'USB']
            
            for card in audio_cards:
                for control in control_names:
                    try:
                        # Try setting volume for each card-control combination
                        cmd = ['amixer', '-c', card, 'set', control, f'{volume_percent}%']
                        result = subprocess.run(cmd, capture_output=True, timeout=3, check=False)
                        if result.returncode == 0:
                            print(f"Set card {card} {control} to {volume_percent}%")
                    except Exception:
                        pass  # Skip if control doesn't exist on this card
            
            # Also try the generic Master control
            subprocess.run(['amixer', 'set', 'Master', f'{volume_percent}%'], timeout=5, check=False)
            
            print(f"Audio volume configuration completed at {volume_percent}%")
        else:
            print(f"Failed to get mixer controls: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Audio volume adjustment timed out")
    except FileNotFoundError:
        print("amixer command not found - audio volume adjustment skipped")
    except Exception as e:
        print(f"Error adjusting audio volume: {e}")

def set_default_audio_device():
    """Set the default audio device to USB Audio (card 0) which is most reliable"""
    try:
        # Create or update ALSA configuration to prefer USB audio
        alsa_conf = """
pcm.!default {
    type hw
    card 0
    device 0
}
ctl.!default {
    type hw
    card 0
}
"""
        
        # Write to user's .asoundrc file
        home_dir = os.path.expanduser("~")
        asoundrc_path = os.path.join(home_dir, ".asoundrc")
        
        with open(asoundrc_path, 'w') as f:
            f.write(alsa_conf)
        
        print(f"Set default audio device to USB Audio (card 0) in {asoundrc_path}")
        
        # Also set PulseAudio default if available
        try:
            subprocess.run(['pactl', 'set-default-sink', '0'], timeout=3, check=False)
            print("Set PulseAudio default sink to device 0")
        except FileNotFoundError:
            print("PulseAudio not available, using ALSA only")
            
    except Exception as e:
        print(f"Error setting default audio device: {e}")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed - CPU monitoring disabled. Install with: pip install psutil")

# Import CPU optimizer
from cpu_optimizer import get_optimizer, optimize_process_priority, enable_cpu_affinity_optimization
from websocket_server import init_websocket_server, update_user_presence

# Argument parser for headless mode
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI display")
# Cam id
parser.add_argument("--camid", type=int, default=1, help="Camera ID")
# Machine id
parser.add_argument("--machineid", type=str, default="7", help="Machine ID")
# Emotion detection mode
parser.add_argument("--emo", action="store_true", help="Enable emotion detection mode")
# LLM gender detection mode
parser.add_argument("--llmgender", action="store_true", help="Use LLM for gender detection instead of InsightFace")
# Audio volume control
parser.add_argument("--vol", type=int, default=100, help="Set audio volume (0-100)")
args = parser.parse_args()

absence_threshold = 5  # seconds
minimum_greeting_interval = 15  # Minimum seconds between greetings

# Speaking and suggestion variables
face_detected = False
suggest_interval = 25  # seconds
stop_event = threading.Event()
is_greeted = False
last_greeting_time = 0  # Track when user was last greeted
greeting_lock = threading.Lock()  # Thread synchronization for greeting logic

# Task monitoring variables
task_monitor = None
application_should_run = True

# Gender detection control
GREET_GENDER_ENABLED = False  # Controls whether gender detection is enabled

# Performance optimization variables
frame_queue = queue.Queue(maxsize=3)  # Small queue to prevent memory buildup
detection_result_queue = queue.Queue(maxsize=5)
latest_frame = None
latest_faces = []
detection_timestamp = 0
frame_lock = threading.Lock()

# Camera optimization flags
ENABLE_FRAME_ROTATION = False  # Set to False to eliminate expensive rotation

# Memory optimization - pre-allocated buffers
frame_buffer_pool = []
BUFFER_POOL_SIZE = 5

# FPS monitoring
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Face tracking for optimization
face_tracking_data = {
    'last_position': None,
    'stability_count': 0,
    'last_detection_time': 0,
    'tracking_mode': False
}

# Frame saving configuration
FRAME_SAVE_DIR = "saved_frames"
ENABLE_FRAME_SAVING = True
FRAME_SAVE_INTERVAL = 30.0  # Save frame every 30 seconds when user present (less frequent for better performance)
FRAME_RETENTION_DAYS = 1  # Delete frames older than 1 day
last_frame_save_time = 0

# WebSocket update timing
last_ws_update_time = 0

# Listening control is now handled by SHOULD_LISTEN Event in speak.py

# Face distance estimation constants
KNOWN_FACE_WIDTH = 0.15  # Average human face width in meters (15cm)
FOCAL_LENGTH = 500  # Approximate focal length, will need calibration for accuracy

# Deck shuffling for auto-suggestions (to avoid repetition)
auto_suggestions_available = []
auto_suggestions_used = []
no_person_suggestions_available = []
no_person_suggestions_used = []
suggestion_lock = threading.Lock()  # Thread safety for suggestion arrays

# Deck shuffling for greetings (to avoid repetition)
greetings_available = []
greetings_used = []
greeting_lock = threading.Lock()  # Thread safety for greeting arrays

# Deck shuffling for busy speak (to avoid repetition)
busy_speak_available = []
busy_speak_used = []
busy_speak_lock = threading.Lock()  # Thread safety for busy speak arrays

def initialize_suggestion_decks():
    """Initialize the deck shuffling arrays for auto-suggestions"""
    global auto_suggestions_available, auto_suggestions_used
    global no_person_suggestions_available, no_person_suggestions_used
    
    with suggestion_lock:
        # Initialize AUTO_SUGGESTIONS deck
        auto_suggestions_available = AUTO_SUGGESTIONS.copy()
        auto_suggestions_used = []
        
        # Initialize NO_PERSON_AUTO_SUGGESTIONS deck
        no_person_suggestions_available = NO_PERSON_AUTO_SUGGESTIONS.copy()
        no_person_suggestions_used = []
        
        print(f"Initialized suggestion decks: {len(auto_suggestions_available)} auto-suggestions, {len(no_person_suggestions_available)} no-person suggestions")

def initialize_greeting_deck():
    """Initialize the deck shuffling arrays for greetings"""
    global greetings_available, greetings_used
    
    with greeting_lock:
        # Initialize GREETINGS deck
        greetings_available = GREETINGs.copy()
        greetings_used = []
        
        print(f"Initialized greeting deck: {len(greetings_available)} greetings")

def get_next_suggestion(is_person_present=True):
    """Get next suggestion using deck shuffling approach"""
    global auto_suggestions_available, auto_suggestions_used
    global no_person_suggestions_available, no_person_suggestions_used
    
    with suggestion_lock:
        if is_person_present:
            # Handle AUTO_SUGGESTIONS
            if not auto_suggestions_available:
                # Refill from used suggestions
                auto_suggestions_available = auto_suggestions_used.copy()
                auto_suggestions_used = []
                print("Refilled auto-suggestions deck")
            
            if auto_suggestions_available:
                # Get random suggestion from available ones
                index = random.randint(0, len(auto_suggestions_available) - 1)
                suggestion = auto_suggestions_available.pop(index)
                auto_suggestions_used.append(suggestion)
                return suggestion
            else:
                return "需要推荐吗?"  # Fallback
        else:
            # Handle NO_PERSON_AUTO_SUGGESTIONS
            if not no_person_suggestions_available:
                # Refill from used suggestions
                no_person_suggestions_available = no_person_suggestions_used.copy()
                no_person_suggestions_used = []
                print("Refilled no-person suggestions deck")
            
            if no_person_suggestions_available:
                # Get random suggestion from available ones
                index = random.randint(0, len(no_person_suggestions_available) - 1)
                suggestion = no_person_suggestions_available.pop(index)
                no_person_suggestions_used.append(suggestion)
                return suggestion
            else:
                return "欢迎光临"  # Fallback

def get_next_greeting():
    """Get next greeting using deck shuffling approach"""
    global greetings_available, greetings_used
    
    with greeting_lock:
        # Handle GREETINGS
        if not greetings_available:
            # Refill from used greetings
            greetings_available = greetings_used.copy()
            greetings_used = []
            print("Refilled greetings deck")
        
        if greetings_available:
            # Get random greeting from available ones
            index = random.randint(0, len(greetings_available) - 1)
            greeting = greetings_available.pop(index)
            greetings_used.append(greeting)
            return greeting
        else:
            return "欢迎光临"  # Fallback

def initialize_busy_speak_deck(busy_speak_array):
    """Initialize the deck shuffling arrays for busy speak"""
    global busy_speak_available, busy_speak_used
    
    with busy_speak_lock:
        # Initialize BUSY_SPEAK deck
        busy_speak_available = busy_speak_array.copy()
        busy_speak_used = []
        
        print(f"Initialized busy speak deck: {len(busy_speak_available)} messages")

def get_next_busy_speak():
    """Get next busy speak message using deck shuffling approach"""
    global busy_speak_available, busy_speak_used
    
    with busy_speak_lock:
        # Handle BUSY_SPEAK
        if not busy_speak_available:
            # Refill from used busy speak messages
            busy_speak_available = busy_speak_used.copy()
            busy_speak_used = []
            print("Refilled busy speak deck")
        
        if busy_speak_available:
            # Get random busy speak message from available ones
            index = random.randint(0, len(busy_speak_available) - 1)
            message = busy_speak_available.pop(index)
            busy_speak_used.append(message)
            return message
        else:
            return "在充电。"  # Fallback

# def getProdcutDetail(machine_id):
    # apiResult = fet

# Speech processing variables - NON-BLOCKING
pending_speech_requests = queue.Queue(maxsize=10)
speech_worker_pool = []

# Task monitoring callbacks
def on_application_start(task_data):
    """Called when application should start running"""
    global application_should_run
    application_should_run = True
    print(f"Application enabled - Task: {task_data.get('currentTaskName')}")

def on_application_stop(task_data):
    """Called when application should stop running"""
    global application_should_run
    application_should_run = False
    print(f"Application disabled - Task Status: {task_data.get('taskStatus')}")
    
    # Clear any ongoing speech safely
    try:
        if NOW_SPEAKING.locked():
            NOW_SPEAKING.release()
    except Exception as e:
        print(f"Note: Lock release handled: {e}")

# Speech worker function for thread pool
def speech_worker(worker_id):
    """Speech worker thread for processing speech requests"""
    print(f"Speech worker {worker_id} started")
    
    while not stop_event.is_set():
        try:
            # Get speech request with timeout
            try:
                speech_request = pending_speech_requests.get(timeout=1.0)
                if speech_request is None:  # Shutdown signal
                    break
            except queue.Empty:
                continue
            
            # Extract speech data
            speech_type = speech_request.get('type', 'text')
            text = speech_request.get('text', '')
            priority = speech_request.get('priority', 1)
            
            if not text:
                continue
            
            # Acquire speaking lock
            if NOW_SPEAKING.acquire(blocking=False):
                try:
                    print(f"Worker {worker_id} speaking: {text[:50]}...")
                    synthesis_text_to_speech_and_play_by_streaming_mode(text=text)
                finally:
                    try:
                        NOW_SPEAKING.release()
                    except Exception as e:
                        print(f"Speech lock release handled: {e}")
            else:
                # If can't acquire lock, put request back with lower priority
                if priority < 3:  # Don't retry more than 3 times
                    retry_request = speech_request.copy()
                    retry_request['priority'] = priority + 1
                    try:
                        pending_speech_requests.put_nowait(retry_request)
                    except queue.Full:
                        print("Speech queue full, dropping request")
            
            # Mark task as done
            pending_speech_requests.task_done()
            
        except Exception as e:
            print(f"Speech worker {worker_id} error: {e}")
            time.sleep(0.1)

# Helper function to queue speech (non-blocking)
def queue_speech(text, speech_type='greeting', priority=1):
    """Queue speech request for processing by worker pool"""
    speech_request = {
        'type': speech_type,
        'text': text,
        'priority': priority,
        'timestamp': time.time()
    }
    
    try:
        pending_speech_requests.put_nowait(speech_request)
        return True
    except queue.Full:
        print("Speech queue full, dropping request")
        return False

# Legacy function for compatibility (now non-blocking)
def play_speech(text):
    """Legacy function - now queues speech instead of blocking"""
    queue_speech(text, 'legacy', priority=2)

# Emotion detection functions
def encode_image(image_data):
    """Encode image data to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_data)
    return base64.b64encode(buffer).decode('utf-8')

def get_emotion_response(frame, products):
    """Get emotion-based response from API"""
    try:
        client = OpenAI(
            api_key="sk-327233dd8f1f4012a7b25283b5da673d",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        print("SAY CHEESE: EMOTION CHEKING,,,,,,,,,")
        time.sleep(1)
        cv2.imwrite(f"./saved_frames/emotion_result_{time.time()}.jpg", frame)
        base64_image = encode_image(frame)
        
        # Create products string from array
        products_str = ",".join(products) if products else "饮品"
        print(f"PRODS: {products_str}")
        # prompt_text = f"根据客户的情绪推荐喝的，一个短句子回答。可推荐的产品：{products_str}. 比如：客户很开心，可以推荐：“来一杯冰美式，提提神 ”"
        prompt_text = f"""
        在图片上的客户的图片中，根据客户当前的情绪，从 {products_str} 中选择一款合适的饮品，用一句有共鸣、富有情感的句子进行推荐。不要总是使用“提提神”，而是根据不同情绪表达不同的语气和用词。还不用说‘从图片来看’直接开始 您看起来
        """
        print(f"PROMPT IMAGE: {prompt_text}")
        
        completion = client.chat.completions.create(
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

def get_llm_gender(frame):
    """Get gender detection from LLM API"""
    try:
        client = OpenAI(
            api_key="sk-327233dd8f1f4012a7b25283b5da673d",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        base64_image = encode_image(frame)
        
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{"role": "user","content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": "这个人是男生还是女生？只回答：男 或者 女"},
                ]}]
        )
        
        gender_result = completion.choices[0].message.content.strip()
        
        # Convert LLM response to the format expected by greet_user
        if "男" in gender_result:
            return 'M'
        elif "女" in gender_result:
            return 'F'
        else:
            return 'unknown'
        
    except Exception as e:
        print(f"LLM Gender API error: {e}")
        return 'unknown'

# Auto-suggest thread function
def suggest_loop():
    if args.emo:
        # Emotion detection mode
        while not stop_event.is_set():
            time.sleep(suggest_interval)
            if application_should_run and face_detected:
                # Get current frame for emotion analysis
                with frame_lock:
                    current_frame = latest_frame
                
                if current_frame is not None:
                    emotion_response = get_emotion_response(current_frame, products)
                    if emotion_response:
                        text = f"。　{emotion_response}　。"
                        queue_speech(text, 'emotion_suggestion', priority=3)
                    else:
                        # Fallback to regular suggestion
                        suggestion = get_next_suggestion(is_person_present=True)
                        text = f"。　{suggestion}　。"
                        queue_speech(text, 'suggestion', priority=3)
    else:
        # Regular suggestion mode
        while not stop_event.is_set():
            time.sleep(suggest_interval)
            if application_should_run and face_detected:
                suggestion = get_next_suggestion(is_person_present=True)
                text = f"。　{suggestion}　。"
                queue_speech(text, 'suggestion', priority=3)  # Lower priority than greetings

# Auto-speak thread function - speaks every minute when no user exists
def auto_speak_loop():
    auto_speak_interval = 60  # 1 minute
    while not stop_event.is_set():
        time.sleep(auto_speak_interval)
        # Only speak when no user is detected
        if application_should_run and not face_detected:
            # Use specific messages for when no user is present
            suggestion = get_next_suggestion(is_person_present=False)
            text = f"。　{suggestion}　。"
            queue_speech(text, 'auto_speak', priority=4)  # Lowest priority

# Charging announcement thread function - announces "charging..." every 3 minutes when application is disabled
def charging_announcement_loop(timeLm,speach):
    charging_interval = timeLm  # 3 minutes (180 seconds)
    while not stop_event.is_set():
        time.sleep(charging_interval)
        # Only announce when application is disabled
        if not application_should_run:
            # Get random busy speak message
            message = get_next_busy_speak()
            queue_speech(message, 'charging', priority=5)  # Very low priority


# Listen status monitoring thread function
def listen_status_monitor():
    """Monitor listening status from API every minute"""
    global GREET_GENDER_ENABLED
    monitor_interval = 60  # 1 minute
    machine_id = args.machineid  # Use machine ID from command line arguments
    
    while not stop_event.is_set():
        try:
            status_result = check_listen_status(machine_id)
            print(f"status_result: {status_result}")
            code = status_result.get("code", 1)
            data = status_result.get("data", False)
            is_greet_gender = data.get("isGreetGender", False) if isinstance(data, dict) else status_result.get("isGreetGender", False)
            
            print(f"Listen Status - Data: {data}, Code: {code}, isGreetGender: {is_greet_gender}")
            
            if code == 1:
                print("Listen Status Error - API returned error code 1")
                # On error, keep current listening state
            else:
                # Update listening status based on API response
                previous_status = SHOULD_LISTEN.is_set()
                if data:
                    SHOULD_LISTEN.set()
                else:
                    SHOULD_LISTEN.clear()
                
                if previous_status != data:
                    status_text = "enabled" if data else "disabled"
                    print(f"Microphone listening {status_text}")
                
                # Update gender detection setting
                previous_gender_setting = GREET_GENDER_ENABLED
                GREET_GENDER_ENABLED = is_greet_gender
                
                if previous_gender_setting != is_greet_gender:
                    gender_text = "enabled" if is_greet_gender else "disabled"
                    print(f"Gender detection {gender_text}")
            
        except Exception as e:
            print(f"Listen Status Monitor Error: {e}")
        
        time.sleep(monitor_interval)


GREETINGs = []

# Greet user based on detected gender (NON-BLOCKING)
def greet_user(gender):
    """Fast, non-blocking greeting function - THREAD-SAFE Ubuntu fix"""
    global is_greeted, GREETINGs, GREET_GENDER_ENABLED, last_greeting_time, greeting_lock
    
    if not GREETINGs:
        return False
    
    # Thread-safe atomic greeting execution (Ubuntu fix)
    current_time = time.time()
    with greeting_lock:
        time_since_last_greeting = current_time - last_greeting_time
        if is_greeted or time_since_last_greeting < minimum_greeting_interval:
            print(f"greet_user() blocked - cooldown active ({minimum_greeting_interval - time_since_last_greeting:.1f}s remaining)")
            return False
        
        # Atomically mark as greeted before processing
        is_greeted = True
        last_greeting_time = current_time
        
    greeting = get_next_greeting()
    if GREET_GENDER_ENABLED and gender == 'M':
        text = f"。先生　{greeting}　。"
    elif GREET_GENDER_ENABLED and gender == 'F':
        text = f"。女士　{greeting}　。"
    else:
        text = f"。　{greeting}　。"
    
    # Queue speech with high priority (greetings are important)
    success = queue_speech(text, 'greeting', priority=1)
    if success:
        print(f"greet_user() success: {text[:20]}...")
    else:
        # If queueing failed, reset the greeting flags
        with greeting_lock:
            is_greeted = False
    return success

# Calculate distance from face width
def calculate_distance(face_width_pixels):
    # Using the formula: distance = (known_width * focal_length) / perceived_width
    distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_pixels
    return distance

def try_camera_indices():
    """Try camera indices 0 and 1 to find a working camera"""
    for cam_id in [0, 1]:
        print(f"Trying camera index {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret:
                print(f"Camera {cam_id} working successfully!")
                return cap, cam_id
            else:
                print(f"Camera {cam_id} opened but can't read frames")
                cap.release()
        else:
            print(f"Camera {cam_id} failed to open")
            cap.release()
    
    return None, -1

def initialize_frame_buffers(width, height):
    """Pre-allocate frame buffers to reduce memory allocation overhead"""
    global frame_buffer_pool
    frame_buffer_pool.clear()
    
    for i in range(BUFFER_POOL_SIZE):
        buffer = np.zeros((height, width, 3), dtype=np.uint8)
        frame_buffer_pool.append(buffer)
    
    print(f"Initialized {BUFFER_POOL_SIZE} frame buffers ({width}x{height})")

def get_frame_buffer():
    """Get a pre-allocated frame buffer"""
    if frame_buffer_pool:
        return frame_buffer_pool.pop()
    else:
        # Fallback if pool is empty
        return None

def return_frame_buffer(buffer):
    """Return frame buffer to pool"""
    if len(frame_buffer_pool) < BUFFER_POOL_SIZE:
        frame_buffer_pool.append(buffer)

def setup_frame_save_directory():
    """Create frame save directory if it doesn't exist"""
    if not os.path.exists(FRAME_SAVE_DIR):
        os.makedirs(FRAME_SAVE_DIR)
        print(f"Created frame save directory: {FRAME_SAVE_DIR}")

def save_frame_with_user(frame, faces):
    """Save frame when user is detected"""
    if not ENABLE_FRAME_SAVING or not faces:
        return
    
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"user_frame_{timestamp}.jpg"
        filepath = os.path.join(FRAME_SAVE_DIR, filename)
        
        # Save frame with moderate compression
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Saved frame: {filename}")
        
    except Exception as e:
        print(f"Error saving frame: {e}")

def cleanup_old_frames():
    """Delete frames older than FRAME_RETENTION_DAYS"""
    try:
        if not os.path.exists(FRAME_SAVE_DIR):
            return
        
        cutoff_time = datetime.now() - timedelta(days=FRAME_RETENTION_DAYS)
        deleted_count = 0
        
        # Find all image files in the save directory
        pattern = os.path.join(FRAME_SAVE_DIR, "user_frame_*.jpg")
        for filepath in glob.glob(pattern):
            try:
                # Get file modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_time:
                    os.remove(filepath)
                    deleted_count += 1
                    
            except Exception as e:
                print(f"Error deleting file {filepath}: {e}")
        
        if deleted_count > 0:
            print(f"Deleted {deleted_count} old frames (older than {FRAME_RETENTION_DAYS} day(s))")
            
    except Exception as e:
        print(f"Error during frame cleanup: {e}")

def parse_gender(face_obj):
    """Parse gender from face detection object, handling different formats"""
    try:
        # Try different possible attribute names
        for attr in ['sex', 'gender', 'g']:
            if hasattr(face_obj, attr):
                gender_value = getattr(face_obj, attr)
                
                # Handle numeric gender (common format: 0=female, 1=male)
                if isinstance(gender_value, (int, float)):
                    return 'M' if gender_value >= 0.5 else 'F'
                
                # Handle string gender
                if isinstance(gender_value, str):
                    gender_lower = gender_value.lower()
                    if gender_lower in ['m', 'male', 'man']:
                        return 'M'
                    elif gender_lower in ['f', 'female', 'woman']:
                        return 'F'
                    
                # Return raw value if it's already in expected format
                if gender_value in ['M', 'F']:
                    return gender_value
        
        # Check if there's a gender prediction array/tensor
        if hasattr(face_obj, 'gender_prob') or hasattr(face_obj, 'gender'):
            print(f"Found gender attributes, debugging...")
            
        return 'unknown'
        
    except Exception as e:
        print(f"Error parsing gender: {e}")
        return 'unknown'

def age_detection(age):
    if age < 18:
        return "小朋友"
    elif age < 25:
        return "年轻人"
    elif age < 35:
        return "中年人"

# Dedicated face detection thread function
def face_detection_worker():
    """Separate thread for face detection processing with intelligent caching"""
    global latest_faces, detection_timestamp, stop_event, application_should_run
    
    # Linux-specific: Track last frame that triggered greeting to prevent rapid duplicates
    last_greeting_frame_time = 0
    
    # Initialize face analysis with minimal modules for speed
    app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
    
    # Get optimizer for performance settings
    optimizer = get_optimizer()
    perf_settings = optimizer.get_performance_settings()
    det_size = (320, 320)  # Use smaller detection size for much faster processing
    app.prepare(ctx_id=-1, det_size=det_size)
    
    # Performance tracking
    detection_times = deque(maxlen=10)  # Track last 10 detection times
    avg_detection_time = 0.2  # Initial estimate
    
    # WebSocket absence tracking
    last_user_present_state = False
    
    print("Face detection worker started")
    
    while not stop_event.is_set():
        try:
            # Check if application should run
            if not application_should_run:
                time.sleep(1)
                continue
            
            # Get frame from queue (with timeout to prevent blocking)
            try:
                frame_data = frame_queue.get(timeout=0.1)
                if frame_data is None:  # Shutdown signal
                    break
                    
                frame, timestamp = frame_data
            except queue.Empty:
                continue
            
            # Perform face detection with timing
            start_time = time.time()
            faces = app.get(frame)
            detection_time = time.time() - start_time
            
            # Update performance tracking
            detection_times.append(detection_time)
            avg_detection_time = sum(detection_times) / len(detection_times)
            
            # Store results globally
            with frame_lock:
                latest_faces = faces
                detection_timestamp = timestamp
            
            # Track user presence state changes for immediate WebSocket updates
            current_user_present = bool(faces)
            if current_user_present != last_user_present_state:
                if not current_user_present:
                    # User just became absent - send immediate update
                    update_user_presence(
                        user_present=False,
                        user_count=0,
                        distance=None,
                        gender=None,
                        age=None
                    )
                last_user_present_state = current_user_present
            
            # Put results in queue for main thread with performance info
            result_data = {
                'faces': faces,
                'timestamp': timestamp,
                'detection_time': detection_time,
                'avg_detection_time': avg_detection_time
            }
            
            # Immediate greeting logic for faster response with proper gender detection
            global is_greeted, GREET_GENDER_ENABLED, last_greeting_time, greeting_lock
            current_time = time.time()
            
            # DISABLED: Move greeting to main loop to fix Linux timing issue
            # The instant greeting was interfering with frame saving and emotion detection on Linux
            
            try:
                detection_result_queue.put_nowait(result_data)
            except queue.Full:
                # Remove old result and add new one
                try:
                    detection_result_queue.get_nowait()
                    detection_result_queue.put_nowait(result_data)
                except queue.Empty:
                    pass
            
            # Mark task as done
            frame_queue.task_done()
            
            # Log performance periodically
            if len(detection_times) == detection_times.maxlen:
                print(f"Face detection avg time: {avg_detection_time:.3f}s")
                detection_times.clear()
            
        except Exception as e:
            print(f"Face detection worker error: {e}")
            time.sleep(0.1)
    
    # When exiting, send final absence update
    update_user_presence(
        user_present=False,
        user_count=0,
        distance=None,
        gender=None,
        age=None
    )

# Frame cleanup thread function
def frame_cleanup_worker():
    """Background thread to periodically clean up old frames"""
    cleanup_interval = 3600  # Check every hour
    
    while not stop_event.is_set():
        try:
            cleanup_old_frames()
            time.sleep(cleanup_interval)
        except Exception as e:
            print(f"Frame cleanup worker error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

# Camera capture loop - OPTIMIZED FOR HIGH PERFORMANCE
def face_detection_loop():
    global face_detected, is_greeted, latest_frame, latest_faces, detection_timestamp
    absent = False
    # Absence timer variables
    absence_start = None
    # Distance threshold timer
    distance_threshold = 1.0  # meters
    distance_far_start = None
    
    # Get CPU optimizer for adaptive performance
    optimizer = get_optimizer()
    
    # Performance optimization variables - ADAPTIVE
    TARGET_FPS = 15  # Higher FPS for camera capture since detection is separate
    frame_time = 1.0 / TARGET_FPS
    last_frame_time = 0
    
    # Detection timing variables
    last_detection_send = 0
    DETECTION_SEND_INTERVAL = 0.5  # Send frames for detection every 500ms (2 FPS) - much slower for better performance
    
    # Adaptive performance update timer
    last_perf_update = 0
    PERF_UPDATE_INTERVAL = 3.0  # Update performance settings every 3 seconds
    
    # CPU monitoring variables
    last_cpu_check = 0
    CPU_CHECK_INTERVAL = 1.5  # Check CPU every 1.5 seconds
    adaptive_frame_skip = 0
    cpu_high_threshold = 70
    cpu_low_threshold = 40
    
    # Frame skipping variables
    skip_frame_count = 0
    FRAME_SKIP_WHEN_ABSENT = 8
    
    # Get initial performance settings
    perf_settings = optimizer.get_performance_settings()
    camera_res = perf_settings['camera_resolution']
    FACE_DETECTION_INTERVAL = perf_settings['face_detection_interval']
    
    # Camera retry loop
    cap = None
    working_cam_id = -1
    camera_retry_interval = 20  # 20 seconds
    last_camera_retry = 0
    
    # Try to initialize camera with retry logic - only try indices 0 and 1
    while cap is None:
        current_time = time.time()
        
        # Only try camera indices 0 and 1
        print("Trying camera indices 0 and 1...")
        cap, working_cam_id = try_camera_indices()
        
        # If still no camera found, wait briefly and retry
        if cap is None:
            print(f"No working camera found (tried indices 0, 1). Retrying immediately...")
            time.sleep(1)  # Brief 1-second delay to prevent excessive CPU usage
            last_camera_retry = current_time
        else:
            print(f"Successfully connected to camera {working_cam_id}")
            break
    
    # Optimize camera settings with ADAPTIVE resolution
    def setup_camera_properties(cap, camera_res, perf_settings):
        """Setup camera properties for optimal performance"""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_res[0])  # Adaptive width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_res[1])  # Adaptive height
        cap.set(cv2.CAP_PROP_FPS, perf_settings['face_detection_fps'])  # Adaptive FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to avoid lag
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus to save CPU
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure processing
    
    setup_camera_properties(cap, camera_res, perf_settings)
    
    # Initialize frame buffers for memory optimization
    initialize_frame_buffers(camera_res[0], camera_res[1])
    
    try:
        while True:
            current_time = time.time()
            
            # Check if application should run
            if not application_should_run:
                face_detected = False
                is_greeted = False
                USER_ABSENT.set()
                if not args.headless:
                    # Show disabled status
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, "APPLICATION DISABLED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.putText(frame, "Task not executing", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("InsightFace", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                time.sleep(1)
                continue
            
            # Update adaptive performance settings
            if current_time - last_perf_update > PERF_UPDATE_INTERVAL:
                perf_settings = optimizer.get_performance_settings()
                TARGET_FPS = perf_settings['face_detection_fps']
                frame_time = 1.0 / TARGET_FPS
                FACE_DETECTION_INTERVAL = perf_settings['face_detection_interval']
                FRAME_SKIP_WHEN_ABSENT = max(8, int(8 * perf_settings['sleep_multiplier']))
                last_perf_update = current_time
            
            # Dynamic CPU monitoring and adaptive frame skipping
            if PSUTIL_AVAILABLE and current_time - last_cpu_check > CPU_CHECK_INTERVAL:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.05)  # Faster CPU check
                    if cpu_percent > cpu_high_threshold:  # High CPU usage
                        adaptive_frame_skip = min(adaptive_frame_skip + 2, 5)  # More aggressive skipping
                        print(f"High CPU detected ({cpu_percent:.1f}%), increasing frame skip to {adaptive_frame_skip}")
                    elif cpu_percent < cpu_low_threshold:  # Low CPU usage
                        adaptive_frame_skip = max(adaptive_frame_skip - 1, 0)
                    last_cpu_check = current_time
                except Exception as e:
                    print(f"CPU monitoring error: {e}")
                    adaptive_frame_skip = 0
            
            # Frame rate limiting with adaptive skipping - OPTIMIZED
            if current_time - last_frame_time < frame_time:
                sleep_time = 0.05 * perf_settings['sleep_multiplier']  # Adaptive sleep time
                time.sleep(sleep_time)
                continue
            last_frame_time = current_time
            
            ret, frame = cap.read()
            if not ret:
                print("Camera disconnected or failed to read frame")
                cap.release()
                
                # Try to reconnect to camera immediately
                print("Attempting to reconnect to camera...")
                last_camera_retry = current_time
                
                # First try the working camera ID
                if working_cam_id != -1:
                    print(f"Trying to reconnect to camera {working_cam_id}...")
                    cap = cv2.VideoCapture(working_cam_id)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret:
                            print(f"Successfully reconnected to camera {working_cam_id}")
                            setup_camera_properties(cap, camera_res, perf_settings)
                            continue
                        else:
                            print(f"Camera {working_cam_id} opened but can't read frames")
                            cap.release()
                
                # If reconnection failed, try camera indices 0 and 1
                print("Trying camera indices 0 and 1...")
                cap, new_cam_id = try_camera_indices()
                if cap is not None:
                    print(f"Successfully switched to camera {new_cam_id}")
                    working_cam_id = new_cam_id
                    setup_camera_properties(cap, camera_res, perf_settings)
                    continue
                else:
                    print("No working camera found (tried indices 0, 1)")
                    cap = None
                
                # If no camera available, wait and continue
                if cap is None:
                    face_detected = False
                    is_greeted = False
                    USER_ABSENT.set()
                    if not args.headless:
                        # Create a black frame to show camera disconnected message
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "CAMERA DISCONNECTED", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.putText(frame, "Retrying continuously...", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow("InsightFace", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    time.sleep(1)
                    continue
                else:
                    time.sleep(1)
                    continue
                
            # Enhanced frame skipping logic - OPTIMIZED FOR FAST USER DETECTION
            # Reduce frame skipping when absent to detect new users faster
            total_skip_frames = max(2, FRAME_SKIP_WHEN_ABSENT // 2) + adaptive_frame_skip  # Reduced skipping
            if absent and skip_frame_count < total_skip_frames:
                skip_frame_count += 1
                sleep_time = 0.05 * perf_settings['sleep_multiplier']  # Reduced sleep time for faster detection
                time.sleep(sleep_time)
                continue
            skip_frame_count = 0
            
            # Store latest frame for display (avoid unnecessary copying)
            with frame_lock:
                latest_frame = frame  # Use reference instead of copy for better performance
            
            # Intelligent detection intervals based on face tracking
            global face_tracking_data
            
            # Determine detection interval based on tracking state
            if face_tracking_data['tracking_mode'] and not absent:
                # When tracking a stable face, reduce detection frequency dramatically
                adaptive_interval = DETECTION_SEND_INTERVAL * 8  # Much slower for stable faces (4 seconds)
            elif absent:
                adaptive_interval = 0.2  # Fast when absent to catch new users
            else:
                adaptive_interval = DETECTION_SEND_INTERVAL  # Normal interval
            
            # Send frame to face detection worker at controlled intervals
            if current_time - last_detection_send >= adaptive_interval:
                # Prepare frame for detection (resize for speed, rotate only if enabled)
                if ENABLE_FRAME_ROTATION:
                    detection_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    detection_frame = frame  # Use frame as-is for better performance
                
                # Resize frame for faster detection (half size = 4x faster processing)
                height, width = detection_frame.shape[:2]
                fast_frame = cv2.resize(detection_frame, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
                
                # Send frame to detection worker (non-blocking)
                try:
                    frame_queue.put_nowait((fast_frame, current_time))
                    last_detection_send = current_time
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
            
            # Get latest detection results (non-blocking)
            faces = []
            try:
                while True:  # Get the most recent result
                    result_data = detection_result_queue.get_nowait()
                    faces = result_data['faces']
                    # Optional: print detection performance
                    # print(f"Detection time: {result_data['detection_time']:.3f}s")
            except queue.Empty:
                # Use cached faces from global variable
                with frame_lock:
                    faces = latest_faces.copy()
            
            face_detected = bool(faces)
            distance_too_far = False
            closest_face_distance = float('inf')
            closest_face_gender = None
            
            # Face tracking and stability analysis
            if face_detected and faces:
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                # Scale face center coordinates to original frame size
                face_center = (((largest_face.bbox[0] + largest_face.bbox[2]) // 2) * 2, 
                              ((largest_face.bbox[1] + largest_face.bbox[3]) // 2) * 2)
                
                if face_tracking_data['last_position'] is not None:
                    # Calculate movement distance
                    movement = np.sqrt((face_center[0] - face_tracking_data['last_position'][0])**2 + 
                                     (face_center[1] - face_tracking_data['last_position'][1])**2)
                    
                    if movement < 50:  # Face is stable (increased threshold for easier tracking)
                        face_tracking_data['stability_count'] += 1
                        if face_tracking_data['stability_count'] > 2:  # Stable for just 2+ frames (faster tracking)
                            face_tracking_data['tracking_mode'] = True
                    else:  # Face moved significantly
                        face_tracking_data['stability_count'] = 0
                        face_tracking_data['tracking_mode'] = False
                
                face_tracking_data['last_position'] = face_center
                face_tracking_data['last_detection_time'] = current_time
            else:
                # No face detected, reset tracking
                face_tracking_data['tracking_mode'] = False
                face_tracking_data['stability_count'] = 0
                if current_time - face_tracking_data['last_detection_time'] > 2.0:
                    face_tracking_data['last_position'] = None

            # Absence detection logic
            if face_detected:
                absence_start = None
                absent = False
                
                # Update WebSocket with user presence (ongoing updates, less frequent than initial detection)
                # This provides continuous updates for existing users
                closest_face_for_ws = None
                if faces:
                    closest_face_for_ws = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                
                # Only send updates every few seconds to avoid spam (initial detection handles immediate updates)
                global last_ws_update_time
                
                if current_time - last_ws_update_time >= 2.0:  # Update every 2 seconds
                    update_user_presence(
                        user_present=True,
                        user_count=len(faces),
                        distance=closest_face_distance if closest_face_distance != float('inf') else None,
                        gender=parse_gender(closest_face_for_ws) if closest_face_for_ws else None,
                        age=getattr(closest_face_for_ws, 'age', None) if closest_face_for_ws else None
                    )
                    last_ws_update_time = current_time
                
                # Process each detected face
                for face in faces:
                    # Scale bounding box coordinates back to original frame size
                    # Face detection was done on half-size frame, so multiply by 2
                    x1, y1, x2, y2 = map(int, face.bbox)
                    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
                    face_width = x2 - x1
                    
                    # Calculate distance in meters
                    distance = calculate_distance(face_width)
                    
                    # Track closest face for greeting
                    if distance < closest_face_distance:
                        closest_face_distance = distance
                        # Use robust gender parsing
                        if args.llmgender and GREET_GENDER_ENABLED:
                            closest_face_gender = get_llm_gender(frame)
                        else:
                            closest_face_gender = parse_gender(face)
                    
                    # Check if distance is too far
                    if distance > distance_threshold:
                        distance_too_far = True
                        if distance_far_start is None:
                            distance_far_start = current_time
                        else:
                            elapsed = current_time - distance_far_start
                            if elapsed >= absence_threshold:
                                absent = True
                            remaining = max(0, absence_threshold - int(elapsed))
                            far_text = f"Too far: {remaining}s"
                            cv2.putText(frame, far_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    else:
                        distance_far_start = None
                    
                    # Batch drawing operations for better performance
                    face_info = {
                        'bbox': (x1, y1, x2, y2),
                        'label': f"{face.sex}:{face.age}",
                        'distance': f"Distance: {distance:.2f}m"
                    }
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Draw labels (use simpler font for performance)
                    cv2.putText(frame, face_info['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, face_info['distance'], (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # FRAME SAVING COMPLETELY DISABLED FOR CPU OPTIMIZATION
                    # Frame saving causes significant I/O overhead and is disabled for performance
                
                # Greeting logic - RE-ENABLED to fix Linux timing issue
                # Only greet if not already greeted to prevent spam
                if not is_greeted:
                    greet_user(closest_face_gender)
                
                # Frame saving logic - save frame when user is present
                global last_frame_save_time
                if (application_should_run and not distance_too_far and 
                    current_time - last_frame_save_time >= FRAME_SAVE_INTERVAL):
                    save_frame_with_user(frame, faces)
                    last_frame_save_time = current_time
                
                # If all faces are not too far, reset the timer
                if not distance_too_far:
                    distance_far_start = None
            else:
                if absence_start is None:
                    absence_start = current_time
                else:
                    elapsed = current_time - absence_start
                    if elapsed >= absence_threshold:
                        absent = True
                    remaining = max(0, absence_threshold - int(elapsed))
                    timer_text = f"User away: {remaining}s"
                    cv2.putText(frame, timer_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if absent:
                cv2.putText(frame, "User not exist", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # Only reset greeting when user becomes absent AND minimum time has passed (thread-safe)
                with greeting_lock:
                    time_since_last_greeting = current_time - last_greeting_time
                    if not is_greeted:  # Avoid setting to False multiple times
                        pass  # Already False
                    elif time_since_last_greeting >= minimum_greeting_interval:
                        is_greeted = False  # Reset greeting only after minimum interval
                        print(f"User absent - greeting reset (after {time_since_last_greeting:.1f}s)")
                    else:
                        # Show remaining time before greeting can reset
                        remaining_time = minimum_greeting_interval - time_since_last_greeting
                        print(f"User absent but greeting cooldown active ({remaining_time:.1f}s remaining)")
                face_detected = False
                if distance_too_far:
                    cv2.putText(frame, "User too far", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                # Set the USER_ABSENT event when user is absent or too far
                USER_ABSENT.set()
                
                # Update WebSocket with user absence
                update_user_presence(
                    user_present=False,
                    user_count=0,
                    distance=None,
                    gender=None,
                    age=None
                )
            else:
                # Clear the USER_ABSENT event when user is present and not too far
                USER_ABSENT.clear()

            # FPS monitoring
            global fps_counter, fps_start_time, current_fps
            fps_counter += 1
            if current_time - fps_start_time >= 5.0:  # Update FPS every 5 seconds
                current_fps = fps_counter / (current_time - fps_start_time)
                print(f"Camera FPS: {current_fps:.1f}")
                fps_counter = 0
                fps_start_time = current_time
            
            if not args.headless:
                # Add performance info to frame
                cv2.putText(frame, f"Camera: {working_cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                tracking_status = "TRACKING" if face_tracking_data['tracking_mode'] else "DETECTING"
                cv2.putText(frame, f"Mode: {tracking_status}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.imshow("InsightFace", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Minimal sleep in headless mode - optimized for fast detection
                if absent:
                    # When user absent, very minimal sleep to detect new users quickly
                    time.sleep(0.01)  # Just 10ms
                else:
                    # When user present, slightly longer but still fast
                    sleep_time = 0.03 * perf_settings['sleep_multiplier']  # Much reduced sleep
                    time.sleep(sleep_time)

    finally:
        stop_event.set()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


def get_user_input():
    mic_listen()
    
# def get_user_input():
#     global userQueryQueue
#     while True:
#         user_input = input("Enter text to synthesize: ")
#         if is_likely_system_echo(user_input, LAST_ASSISTANT_RESPONSE):
#             print("Detected system echo, skipping response")
#             continue
#         userQueryQueue.put(user_input)

if __name__ == "__main__":
    # print(NO_RESPONSE_NEEDED_RULE)
    # global AUTO_SUGGESTIONSs
    # global GREETINGs
    # from aiUnderstandPrompt import PromptUnderstand
    from listener import set_application_state_reference
    
    # Configure audio system before starting the application
    print("Setting up audio system...")
    set_default_audio_device()
    increase_audio_volume(args.vol)
    
    # Initialize CPU optimizations
    print("Initializing CPU optimizations...")
    optimize_process_priority()
    enable_cpu_affinity_optimization()
    
    # Start CPU monitoring
    cpu_optimizer = get_optimizer()
    cpu_optimizer.start_monitoring()
    
    # Initialize WebSocket server
    print("Initializing WebSocket server...")
    # websocket_server = init_websocket_server(host='0.0.0.0', port=8765)
    
    # Initialize TTS
    print("Initializing TTS API...")
    init_dashscope_api_key()    

    print("Fetching product data from API...")
    try:
        api_result = fetch_product_by_name(args.machineid)
        if 'error' in api_result:
            print(f"API Error: {api_result['error']}")
            print("Using fallback configuration...")
            result = None
        else:
            result = api_result['data']
            print("Product data loaded successfully.")
    except Exception as e:
        print(f"Failed to fetch API data: {e}")
        print("Using fallback configuration...")
        result = None
    # print(f"RES:::: {result}")
    # sys.exit(0)
    # Initialize configuration with defaults from chat.py
    print("Initializing configuration with defaults and API data...")
    
    # Use default values as fallback, override with API data if available
    products = result.get("products") if result else default.get("products", ["盲盒"])
    prompt = result.get("prompt") if result else default.get("prompt", "你是一个友好的咖啡店助手。")
    GREETINGs = result.get("greetings") if result else default.get("greetings", ["欢迎光临", "您好", "欢迎"])
    AUTO_SUGGESTIONS = result.get("suggestions") if result else default.get("suggestions", ["需要推荐吗?", "要试试我们的招牌饮品吗?", "有什么可以帮您的?"])
    NO_PERSON_AUTO_SUGGESTIONS = result.get("noPersonSuggestions") if result else default.get("noPersonSuggestions", ["欢迎光临", "需要帮助请随时呼唤我", "今日特色等您品尝"])
    busy_speak = result.get("busySpeak") if result else default.get("busySpeak", ["在充电。"])
    busy_speak_time = result.get("busySpeakTime", 180) if result else default.get("busySpeakTime", "180")
    
    # Initialize gender detection setting
    GREET_GENDER_ENABLED = result.get("isGreetGender") if result else default.get("isGreetGender", False)
    
    # Ensure busy_speak_time is integer
    if isinstance(busy_speak_time, str):
        busy_speak_time = int(busy_speak_time)
    
    # Build system prompt
    if products:
        prompt += ".你可以推荐以下饮品：\n" + ",".join(products)    
        NO_RESPONSE_NEEDED_RULE = "。\n重要过滤指令: 如果对话不是关于'"  + ",".join(products)+ "' "+  NO_RESPONSE_NEEDED_RULE
        prompt += NO_RESPONSE_NEEDED_RULE
    
    SYSTEM_PROMPT = prompt
    
    # Print configuration source
    if result:
        print("Using API configuration with defaults as fallback")
    else:
        print("Using default configuration from chat.py")
    
    print(f"Products: {products}")
    print(f"Greetings: {len(GREETINGs)} items")
    print(f"Suggestions: {len(AUTO_SUGGESTIONS)} items")
    print(f"No-person suggestions: {len(NO_PERSON_AUTO_SUGGESTIONS)} items")
    print(f"Busy speak: {len(busy_speak)} items")
    print(f"Busy speak time: {busy_speak_time}s")
    print(f"Gender detection: {'enabled' if GREET_GENDER_ENABLED else 'disabled'}")
    
    # Initialize deck shuffling for all components
    initialize_suggestion_decks()
    initialize_greeting_deck()
    initialize_busy_speak_deck(busy_speak)
    
    # Initialize task monitoring
    print("Initializing task monitoring...")
    task_monitor = TaskMonitor(machine_id=args.machineid)
    task_monitor.set_application_callbacks(on_application_start, on_application_stop)
    
    # Set application state reference for listener
    set_application_state_reference(lambda: application_should_run)
    
    # Setup frame saving
    setup_frame_save_directory()
    
    # Start task monitoring
    task_monitor.start_monitoring()
    
    print("Starting application threads...")
    
    # Start speech worker pool (2 workers for parallel speech processing)
    for i in range(2):
        worker_thread = threading.Thread(target=speech_worker, args=(i,), daemon=True)
        worker_thread.start()
        speech_worker_pool.append(worker_thread)
    
    threading.Thread(target=face_detection_worker, daemon=True).start()  # Start face detection worker
    threading.Thread(target=face_detection_loop, daemon=True).start()    # Start camera capture loop
    threading.Thread(target=frame_cleanup_worker, daemon=True).start()   # Start frame cleanup worker
    threading.Thread(target=LLM_Speak, args=(SYSTEM_PROMPT,), daemon=True).start()
    threading.Thread(target=suggest_loop, daemon=True).start()
    threading.Thread(target=auto_speak_loop, daemon=True).start()
    threading.Thread(target=listen_status_monitor, daemon=True).start()
    threading.Thread(target=charging_announcement_loop, args=(int(busy_speak_time),busy_speak,), daemon=True).start()  # Start charging announcement thread
    threading.Thread(target=mic_listen, daemon=True).start()
    # threading.Thread(target=get_user_input, daemon=True).start()
    
    print("Application started successfully!")
    
    # Prevent main thread from exiting
    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        # Cleanup task monitoring
        if task_monitor:
            task_monitor.stop_monitoring()
        
        # Cleanup frame queues
        try:
            frame_queue.put_nowait(None)  # Signal face detection worker to stop
        except queue.Full:
            pass
        
        # Cleanup speech workers
        for _ in speech_worker_pool:
            try:
                pending_speech_requests.put_nowait(None)  # Signal speech workers to stop
            except queue.Full:
                pass