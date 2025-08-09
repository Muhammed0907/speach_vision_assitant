import cv2
import time
import threading
import argparse
from insightface.app import FaceAnalysis
from speak import init_dashscope_api_key, synthesis_text_to_speech_and_play_by_streaming_mode,LLM_Speak,userQueryQueue,LAST_ASSISTANT_RESPONSE,STOP_EVENT,NOW_SPEAKING
from greetings import male_greetings, female_greetings, neutral_greetings
from suggestion import AUTO_SUGGESTIONS
from chat import SYSTEM_PROMPT
from echocheck import is_likely_system_echo
import random

# Seed random number generator for better randomness
random.seed()

# Argument parser for headless mode
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI display")
args = parser.parse_args()

# Initialize face analysis
app = FaceAnalysis(allowed_modules=['detection','genderage'])
app.prepare(ctx_id=-1, det_size=(640, 640))

cap = cv2.VideoCapture(0)

# Absence timer variables
absence_start = None
absent = False
absence_threshold = 5  # seconds

# Speaking and suggestion variablesace_detected = False
suggest_interval = 10  # seconds
stop_event = threading.Event()
# NOW_SPEAKING = threading.Lock()
is_greeted = False

# Initialize TTS
init_dashscope_api_key()

# Helper to play speech in thread and clear LOCK when done
def play_speech(text):
    try:
        synthesis_text_to_speech_and_play_by_streaming_mode(text=text)
    finally:
        NOW_SPEAKING.release()

# Auto-suggest thread function
def suggest_loop():
    while not stop_event.is_set():
        time.sleep(suggest_interval)
        if face_detected and NOW_SPEAKING.acquire(blocking=False):
            index = random.randint(0, len(AUTO_SUGGESTIONS) - 1)
            t = threading.Thread(target=play_speech, args=(f"。{AUTO_SUGGESTIONS[index]}。",), daemon=True)
            t.start()

# Start auto-suggest thread
auto_thread = threading.Thread(target=suggest_loop, daemon=True)
auto_thread.start()

threading.Thread(target=LLM_Speak, args=(SYSTEM_PROMPT,), daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get faces from original frame - let InsightFace handle resizing internally
        faces = app.get(frame)
        current_time = time.time()
        face_detected = bool(faces)

        # Greeting logic
        if face_detected and not is_greeted:
            if NOW_SPEAKING.acquire(blocking=False):
                # Import parse_gender function from main
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from main import parse_gender
                
                # choose greeting based on detected gender - using robust parsing
                gender = parse_gender(faces[0])
                print(f"insightfacePyWith.py - Detected gender: {gender}")
                
                if gender == 'M':
                    lst = male_greetings
                elif gender == 'F':
                    lst = female_greetings
                else:
                    lst = neutral_greetings
                text = f"。{random.choice(lst)}。"
                t = threading.Thread(target=play_speech, args=(text,), daemon=True)
                t.start()
                is_greeted = True

        try:
            user_input = input("Enter text to synthesize: ")
            if is_likely_system_echo(user_input, LAST_ASSISTANT_RESPONSE):
                print("Detected system echo, skipping response")
                # continue
            else:  
                stop_event.set() 
                STOP_EVENT.set()          # signal “stop current speech”
                userQueryQueue.put(user_input)   # hand off to the single LLM_Speak loop
        except KeyboardInterrupt:
            print("Stopping on user interrupt…")
            break


        # Absence detection logic
        if face_detected:
            absence_start = None
            absent = False
            # Draw faces (bounding boxes are already in original frame coordinates)
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # Use robust gender parsing for display too
                display_gender = parse_gender(face)
                label = f"{display_gender}:{face.age}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        else:
            if absence_start is None:
                absence_start = current_time
            else:
                elapsed = current_time - absence_start
                if elapsed >= absence_threshold:
                    absent = True
                remaining = max(0, absence_threshold - int(elapsed))
                timer_text = f"User away: {remaining}s"
                cv2.putText(frame, timer_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

        if absent:
            cv2.putText(frame, "User not exist", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
            is_greeted = False

        # Display
        if not args.headless:
            cv2.imshow("InsightFace", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.01)

finally:
    stop_event.set()
    auto_thread.join()
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

