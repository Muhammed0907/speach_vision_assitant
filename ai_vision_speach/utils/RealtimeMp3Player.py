#!/usr/bin/env python3
# Copyright (C) Alibaba Group. All Rights Reserved.
# MIT License (https://opensource.org/licenses/MIT)

import subprocess
import threading

import pyaudio


# Define a callback to handle the result
class RealtimeMp3Player:
    def __init__(self, verbose=False):
        self.ffmpeg_process = None
        self._stream = None
        self._player = None
        self.play_thread = None
        self.stop_event = threading.Event()
        self.verbose = verbose
        self.working_sample_rate = 22050  # Default sample rate

    def reset(self):
        self.ffmpeg_process = None
        self._stream = None
        self._player = None
        self.play_thread = None
        self.stop_event = threading.Event()
        self.working_sample_rate = 22050  # Reset to default

    def start(self):
        try:
            self._player = pyaudio.PyAudio()  # initialize pyaudio to play audio
            
            # Try to find a working audio device
            device_index = self._find_audio_device()
            
            # Try to open audio stream with error handling
            try:
                self._stream = self._player.open(
                    format=pyaudio.paInt16, 
                    channels=1, 
                    rate=self.working_sample_rate,
                    output=True,
                    output_device_index=device_index)  # initialize pyaudio stream
            except OSError as e:
                if "Unanticipated host error" in str(e):
                    print(f"Audio device error: {e}. Trying to recover...")
                    # Clean up and retry with different approach
                    try:
                        self._player.terminate()
                        self._player = pyaudio.PyAudio()
                        # Force use of ALSA directly if PulseAudio fails
                        self._stream = self._player.open(
                            format=pyaudio.paInt16, 
                            channels=1, 
                            rate=self.working_sample_rate,
                            output=True,
                            output_device_index=0)  # Force device 0 (bcm2835)
                        print("Successfully recovered using ALSA device 0")
                    except Exception as retry_error:
                        print(f"Recovery failed: {retry_error}")
                        raise
                else:
                    raise
            
            try:
                self.ffmpeg_process = subprocess.Popen(
                    [
                        'ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', str(self.working_sample_rate),
                        '-ac', '1', 'pipe:1'
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )  # initialize ffmpeg to decode mp3
                if self.verbose:
                    print(f'mp3 audio player is started with sample rate: {self.working_sample_rate} Hz')
            except subprocess.CalledProcessError as e:
                # Capturing ffmpeg exceptions, printing error details
                print(f'FFmpeg error: {e}')
                raise
        except Exception as e:
            print(f'Audio initialization error: {e}')
            raise
    
    def _find_audio_device(self):
        """Find a working audio output device, preferring USB audio"""
        # Priority order: USB Audio first, then others
        device_priorities = []
        
        # Collect all available devices with their info
        for i in range(self._player.get_device_count()):
            try:
                device_info = self._player.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    name = device_info['name'].lower()
                    if self.verbose:
                        print(f"Found audio device {i}: {device_info['name']}")
                    
                    # Prioritize USB audio devices
                    if 'usb' in name or 'v-110' in name:
                        device_priorities.insert(0, (i, device_info))  # High priority
                    else:
                        device_priorities.append((i, device_info))  # Normal priority
            except Exception as e:
                if self.verbose:
                    print(f"Error checking device {i}: {e}")
                continue
        
        # Try devices in priority order with multiple sample rates
        sample_rates = [22050, 44100, 48000, 16000, 8000]  # Common sample rates
        
        for device_index, device_info in device_priorities:
            if self.verbose:
                print(f"Trying audio device {device_index}: {device_info['name']}")
            
            # Try different sample rates for this device
            for rate in sample_rates:
                try:
                    # Test if this device works with this sample rate
                    test_stream = self._player.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        output=True,
                        output_device_index=device_index,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    
                    if self.verbose:
                        print(f"Successfully using audio device {device_index}: {device_info['name']} at {rate} Hz")
                    
                    # Store the working sample rate for later use
                    self.working_sample_rate = rate
                    return device_index
                    
                except Exception as e:
                    if self.verbose and rate == sample_rates[0]:  # Only show error for first rate
                        print(f"Device {device_index} failed with rate {rate}: {e}")
                    continue
            
            if self.verbose:
                print(f"Device {device_index} failed with all sample rates")
        
        # If no device works, return None and let PyAudio handle it
        print("Warning: No working audio device found, using PyAudio default")
        return None

    def stop(self):
        try:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.play_thread.join()
            self._stream.stop_stream()
            self._stream.close()
            self._player.terminate()
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
            if self.verbose:
                print('mp3 audio player is stopped')
        except subprocess.CalledProcessError as e:
            # Capturing ffmpeg exceptions, printing error details
            print(f'An error occurred: {e}')

    def play_audio(self):
        # play audio with pcm data decode by ffmpeg
        try:
            while not self.stop_event.is_set():
                pcm_data = self.ffmpeg_process.stdout.read(1024)
                if pcm_data:
                    self._stream.write(pcm_data)
                else:
                    break
        except subprocess.CalledProcessError as e:
            # Capturing ffmpeg exceptions, printing error details
            print(f'An error occurred: {e}')

    def write(self, data: bytes) -> None:
        # print('write audio data:', len(data))
        try:
            self.ffmpeg_process.stdin.write(data)
            if self.play_thread is None:
                # initialize play thread
                # print('start play thread')
                self._stream.start_stream()
                self.play_thread = threading.Thread(target=self.play_audio)
                self.play_thread.start()
        except subprocess.CalledProcessError as e:
            # Capturing ffmpeg exceptions, printing error details
            print(f'An error occurred: {e}')