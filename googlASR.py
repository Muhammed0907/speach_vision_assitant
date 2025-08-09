from google import genai
from google.genai import types
import pygame
import io
import wave


# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)
      
client = genai.Client(api_key="AIzaSyBoAEFmLXyF75KhrXO9twwBjCguArL6JQs")

response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents="Say cheerfully: Have a wonderful day!",
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore',
            )
         )
      ),
   )
)

data = response.candidates[0].content.parts[0].inline_data.data
file_name='out1.wav'

wave_file(file_name, data) # Saves the file to current directory

# Initialize pygame mixer for audio playback
pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)

# Create a wave file in memory
buffer = io.BytesIO()
with wave.open(buffer, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(data)

buffer.seek(0)

# Load and play the audio
pygame.mixer.music.load(buffer)
pygame.mixer.music.play()

# Wait for the audio to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.wait(100)

