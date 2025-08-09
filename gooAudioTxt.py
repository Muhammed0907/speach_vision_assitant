import base64
import pyaudio
import wave
import threading
import time
import io
from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyBoAEFmLXyF75KhrXO9twwBjCguArL6JQs",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class RealtimeAudioTranscriber:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 3
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        
    def record_audio_chunk(self):
        stream = self.audio.open(format=self.format,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                frames_per_buffer=self.chunk)
        
        print("Recording...")
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        print("Recording finished")
        stream.stop_stream()
        stream.close()
        
        # Save to bytes buffer
        wav_buffer = io.BytesIO()
        wf = wave.open(wav_buffer, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        wav_buffer.seek(0)
        return wav_buffer.getvalue()
    
    def transcribe_audio(self, audio_data):
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        }
                    ],
                }
            ],
        )
        
        return response.choices[0].message.content
    
    def start_realtime_transcription(self):
        print("Starting real-time transcription. Press Ctrl+C to stop.")
        self.is_recording = True
        
        try:
            while self.is_recording:
                audio_data = self.record_audio_chunk()
                transcription = self.transcribe_audio(audio_data)
                print(f"Transcription: {transcription}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nStopping transcription...")
            self.is_recording = False
        finally:
            self.audio.terminate()

if __name__ == "__main__":
    transcriber = RealtimeAudioTranscriber()
    transcriber.start_realtime_transcription()