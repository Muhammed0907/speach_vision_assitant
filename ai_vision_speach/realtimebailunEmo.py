import os
import time
import base64
import cv2
from openai import OpenAI
from datetime import datetime
import insightface
import threading
import queue

def encode_image(image_data):
    """Encode image data to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_data)
    return base64.b64encode(buffer).decode('utf-8')

def save_frame(frame, frame_count):
    """Save frame to saved_frames directory"""
    os.makedirs("saved_frames", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_frames/frame_{frame_count}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def process_frame_worker(frame_queue, client, app, stop_event):
    """Worker thread for processing frames"""
    while not stop_event.is_set():
        try:
            frame_data = frame_queue.get(timeout=1)
            if frame_data is None:
                break
                
            frame, frame_count = frame_data
            
            faces = app.get(frame)
            
            if len(faces) == 0:
                print(f"Frame {frame_count}: No person detected, skipping API call")
            else:
                print(f"Frame {frame_count}: {len(faces)} person(s) detected")
                
                saved_filename = save_frame(frame, frame_count)
                print(f"Frame saved as {saved_filename}")
                
                encode_start = time.time()
                base64_image = encode_image(frame)
                encode_end = time.time()
                print(f"Image encoding took: {encode_end - encode_start:.3f} seconds")
                
                api_start = time.time()
                try:
                    completion = client.chat.completions.create(
                        model="qwen-vl-max",
                        messages=[{"role": "user","content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": "根据客户的情绪推荐吃的，一个短句子回答"},
                            # {"type": "text", "text": "ta是男生还是女生,用一字"},
                            ]}]
                    )
                    api_end = time.time()
                    print(f"API call took: {api_end - api_start:.3f} seconds")
                    
                    emotion_result = completion.choices[0].message.content
                    print(f"Detected emotion: {emotion_result}")
                    
                except Exception as e:
                    print(f"API error: {e}")
                    
                print("-" * 50)
            
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker thread error: {e}")

def main():
    client = OpenAI(
        api_key="sk-327233dd8f1f4012a7b25283b5da673d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    app = insightface.app.FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    
    worker_thread = threading.Thread(
        target=process_frame_worker,
        args=(frame_queue, client, app, stop_event)
    )
    worker_thread.daemon = True
    worker_thread.start()
    
    print("Starting real-time emotion detection...")
    print("Press 'q' to quit")
    
    frame_count = 0
    last_process_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera!")
                break
            
            frame_count += 1
            cv2.imshow('Camera Feed', frame)
            
            current_time = time.time()
            if current_time - last_process_time >= 3.0:
                if not frame_queue.full():
                    frame_queue.put((frame.copy(), frame_count))
                    last_process_time = current_time
                else:
                    print("Processing queue full, skipping frame")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping emotion detection...")
    finally:
        stop_event.set()
        frame_queue.put(None)
        worker_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        print("Camera resources cleaned up.")

if __name__ == "__main__":
    main()