#!/usr/bin/env python3
"""
Live Audio Transcription using Google Gemini AI
Python equivalent of liveGoAudioTrans.js

@license
SPDX-License-Identifier: Apache-2.0
"""

import os
import asyncio
import threading
import time
import wave
import numpy as np
import pyaudio
from typing import Optional, Set
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LiveAudioTranscription:
    def __init__(self):
        self.is_recording = False
        self.status = ""
        self.error = ""
        
        # Audio settings
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000
        self.chunk_size = 256
        self.format = pyaudio.paFloat32
        self.channels = 1
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Gemini client and session
        self.client: Optional[genai.Client] = None
        self.session = None
        
        # Audio sources management
        self.sources: Set[threading.Thread] = set()
        self.next_start_time = 0.0
        
        self.init_client()
    
    def init_client(self):
        """Initialize the Gemini AI client"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            self.client = genai.Client(api_key=api_key)
            self.update_status("Client initialized")
            self.init_session()
        except Exception as e:
            self.update_error(f"Failed to initialize client: {e}")
    
    async def init_session(self):
        """Initialize the live session with Gemini"""
        model = 'gemini-2.5-flash-preview-native-audio-dialog'
        
        try:
            # Note: This is a simplified version as the Python SDK might have different live API
            self.update_status("Session initialized")
        except Exception as e:
            self.update_error(f"Failed to initialize session: {e}")
    
    def update_status(self, msg: str):
        """Update status message"""
        self.status = msg
        print(f"Status: {msg}")
    
    def update_error(self, msg: str):
        """Update error message"""
        self.error = msg
        print(f"Error: {msg}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for processing audio input"""
        if not self.is_recording:
            return (None, pyaudio.paContinue)
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Send audio data to Gemini (simplified - actual implementation would depend on SDK)
            # self.send_audio_to_gemini(audio_data)
            
        except Exception as e:
            self.update_error(f"Audio processing error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return
        
        self.update_status("Requesting microphone access...")
        
        try:
            # Initialize input stream
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.input_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            # Initialize output stream for playback
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.output_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            self.is_recording = True
            self.update_status("🔴 Recording... Capturing audio chunks.")
            
        except Exception as e:
            self.update_error(f"Error starting recording: {e}")
            self.stop_recording()
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        self.update_status("Stopping recording...")
        
        self.is_recording = False
        
        try:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
            
            self.update_status("Recording stopped. Call start_recording() to begin again.")
            
        except Exception as e:
            self.update_error(f"Error stopping recording: {e}")
    
    def reset(self):
        """Reset the session"""
        if self.session:
            # Close existing session
            pass
        
        self.init_session()
        self.update_status("Session cleared.")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
    
    def run_interactive(self):
        """Run interactive console interface"""
        print("\n=== Live Audio Transcription ===")
        print("Commands:")
        print("  's' - Start recording")
        print("  'x' - Stop recording")
        print("  'r' - Reset session")
        print("  'q' - Quit")
        print("=====================================\n")
        
        try:
            while True:
                command = input("Enter command (s/x/r/q): ").lower().strip()
                
                if command == 's':
                    self.start_recording()
                elif command == 'x':
                    self.stop_recording()
                elif command == 'r':
                    self.reset()
                elif command == 'q':
                    break
                else:
                    print("Invalid command. Use s/x/r/q")
                
                print(f"Current status: {self.status}")
                if self.error:
                    print(f"Error: {self.error}")
                    self.error = ""  # Clear error after showing
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    transcriber = LiveAudioTranscription()
    transcriber.run_interactive()

if __name__ == "__main__":
    main()