import pyaudio
import numpy as np
import threading
import queue
import time
from faster_whisper import WhisperModel

class RealtimeWhisper:
    def __init__(self, model_size="tiny"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 3  # Process every 3 seconds
        
    def start_recording(self):
        self.is_recording = True
        audio = pyaudio.PyAudio()
        
        print(f"Available audio devices: {audio.get_device_count()}")
        
        stream = audio.open(format=self.format,
                           channels=self.channels,
                           rate=self.rate,
                           input=True,
                           frames_per_buffer=self.chunk)
        
        print("Recording... Press Ctrl+C to stop")
        print("Audio stream opened successfully")
        
        try:
            while self.is_recording:
                frames = []
                for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                    if not self.is_recording:
                        break
                    data = stream.read(self.chunk)
                    frames.append(data)
                
                if frames and self.is_recording:
                    audio_data = b''.join(frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    print(f"Captured {len(audio_np)} audio samples")
                    self.audio_queue.put(audio_np)
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            self.is_recording = False
    
    def process_audio(self):
        print("Audio processing thread is running...")
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=2)
                print(f"Got audio data from queue: {len(audio_data)} samples")
                
                # Check if audio has sufficient volume
                audio_level = np.sqrt(np.mean(audio_data ** 2))
                print(f"Audio level: {audio_level:.4f}")
                
                if audio_level > 0.001:  # Minimum threshold
                    print("Processing audio...")
                    segments, info = self.model.transcribe(audio_data, beam_size=1)
                    
                    for segment in segments:
                        if segment.text.strip():
                            print(f"[{time.strftime('%H:%M:%S')}] {segment.text.strip()}")
                        else:
                            print("Empty transcription segment")
                else:
                    print("Audio too quiet, skipping...")
                
            except queue.Empty:
                if not self.is_recording:
                    break
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
        print("Audio processing thread finished")
    
    def start(self):
        # Start audio processing thread
        print("Starting audio processing thread...")
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
        print("Audio processing thread started")
        
        # Start recording (blocking)
        self.start_recording()

if __name__ == "__main__":
    whisper_rt = RealtimeWhisper()
    whisper_rt.start()