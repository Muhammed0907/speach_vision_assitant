# For prerequisites running the following sample, visit https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen
import os
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording and translation)
import sys
import time
import threading
import random  # For implementing backoff strategy
import aiohttp  # For aiohttp exception handling
import pyaudio  # For pyaudio error handling

import dashscope
from dashscope.audio.asr import *
from speak import userQueryQueue, LAST_ASSISTANT_RESPONSE, NOW_SPEAKING, USER_ABSENT, SHOULD_LISTEN
from echocheck import is_likely_system_echo

# Global application state - will be set by main.py
APPLICATION_SHOULD_RUN = None

def set_application_state_reference(state_ref):
    """Set reference to application state"""
    global APPLICATION_SHOULD_RUN
    APPLICATION_SHOULD_RUN = state_ref

mic = None
stream = None

# ULTRA optimized recording parameters for maximum CPU efficiency
sample_rate = 16000  # sampling rate (Hz) - keeping standard for compatibility
channels = 1  # mono channel
dtype = 'int16'  # data type
format_pcm = 'pcm'  # the format of the audio data
block_size = 9600  # Increased from 6400 to 9600 to further reduce processing frequency (600ms chunks)

# Maximum reconnection attempts before giving up
MAX_RETRY_ATTEMPTS = 10
# Initial backoff time in seconds
INITIAL_BACKOFF = 2
# Maximum time to wait for a recognition response
RECOGNITION_TIMEOUT = 15  # seconds

def init_dashscope_api_key():
    """
        Set your DashScope API-key. More information:
        https://github.com/aliyun/alibabacloud-bailian-speech-demo/blob/master/PREREQUISITES.md
    """

    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ[
            'DASHSCOPE_API_KEY']  # load API-key from environment variable DASHSCOPE_API_KEY
    else:
        dashscope.api_key = '<your-dashscope-api-key>'  # set API-key manually

# Real-time speech recognition callback
class Callback(RecognitionCallback):
    def on_open(self) -> None:
        global mic, stream
        print('RecognitionCallback open.')
        try:
            # Only create a new PyAudio instance if one doesn't exist
            if mic is None:
                mic = pyaudio.PyAudio()
                
            # Always create a fresh stream on open to ensure it's in a good state
            # First clean up any existing stream
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
                    
            # Create a new stream with optimized buffer size
            stream = mic.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=block_size)  # Use the optimized block_size
            print("New audio stream created")
        except Exception as e:
            print(f"Error initializing audio: {e}")
            # Try to reset the audio system
            self._reset_audio()
            
    def _reset_audio(self):
        """Reset the audio system completely"""
        global mic, stream
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
                stream = None
                
            if mic is not None:
                try:
                    mic.terminate()
                except:
                    pass
                mic = None
                
            # Create fresh instances
            time.sleep(1)  # Brief pause to let the system recover
            mic = pyaudio.PyAudio()
            stream = mic.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=block_size)  # Use the optimized block_size
            print("Audio system completely reset")
        except Exception as e:
            print(f"Failed to reset audio system: {e}")
            # Will try again on next reconnection

    def on_close(self) -> None:
        global mic, stream
        print('RecognitionCallback close.')
        # Only close the stream, don't terminate PyAudio to avoid reopening issues
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        stream = None
        # We'll keep mic object alive for reuse

    def on_complete(self) -> None:
        print('RecognitionCallback completed.')  # translation completed

    def on_error(self, message) -> None:
        print('RecognitionCallback task_id: ', message.request_id)
        print('RecognitionCallback error: ', message.message)
        # Stop and close the audio stream if it is running
        global stream, mic
        if stream is not None and hasattr(stream, 'is_active') and stream.is_active():
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        
        # Don't exit the program, just print the error
        print("Speech recognition error occurred. Will try to reconnect.")

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        global userQueryQueue
        if 'text' in sentence:
            if RecognitionResult.is_sentence_end(sentence):
                print('RecognitionCallback text: ', sentence['text'])
                        # Check if system is speaking (lock is acquired)
                if NOW_SPEAKING.locked():
                    print("System is speaking, ignoring user input")
                    return
                # Check if application should run
                if APPLICATION_SHOULD_RUN is not None and not APPLICATION_SHOULD_RUN():
                    print("Application disabled, ignoring user input")
                    return
                # Also check for echo
                if is_likely_system_echo(sentence['text'], LAST_ASSISTANT_RESPONSE):
                    print("Detected system echo, skipping response")
                    return
                userQueryQueue.put(sentence['text'])
                print(
                    'RecognitionCallback sentence end, request_id:%s, usage:%s'
                    % (result.get_request_id(), result.get_usage(sentence)))


def signal_handler(sig, frame):
    print('Ctrl+C pressed, stop translation ...')
    # Stop translation
    recognition.stop()
    print('Translation stopped.')
    print(
        '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
        .format(
            recognition.get_last_request_id(),
            recognition.get_first_package_delay(),
            recognition.get_last_package_delay(),
        ))
    # Forcefully exit the program
    sys.exit(0)


# def isLenedteEntd

def mic_listen():
    global mic, stream
    callback = Callback()
    recognition = None
    retry_count = 0
    last_activity_time = time.time()
    consecutive_recognition_stops = 0  # Track consecutive recognition stops
    recognition_paused_for_speech = False
    recognition_paused_for_absence = False
    recognition_paused_for_listen_status = False
    last_check_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            
            # FIRST CHECK: If application should not run, skip everything
            if APPLICATION_SHOULD_RUN is not None and not APPLICATION_SHOULD_RUN():
                # Stop any existing recognition
                if recognition is not None:
                    try:
                        recognition.stop()
                        print("Recognition service stopped - application disabled")
                    except Exception as e:
                        print(f"Error stopping recognition: {e}")
                    recognition = None
                
                # Reset all pause flags
                recognition_paused_for_listen_status = True
                recognition_paused_for_speech = False
                recognition_paused_for_absence = False
                
                # Wait longer to reduce CPU usage when application is disabled
                time.sleep(3.0)
                continue
            
            # SECOND CHECK: If listening is disabled by API, skip everything
            if not SHOULD_LISTEN.is_set():
                # Stop any existing recognition
                if recognition is not None:
                    try:
                        recognition.stop()
                        print("Recognition service stopped due to API setting")
                    except Exception as e:
                        print(f"Error stopping recognition: {e}")
                    recognition = None
                
                # Reset all pause flags
                recognition_paused_for_listen_status = True
                recognition_paused_for_speech = False
                recognition_paused_for_absence = False
                
                # Wait longer to reduce CPU usage when listening is disabled
                time.sleep(3.0)
                continue
            
            # ULTRA OPTIMIZED: Maximum status check frequency reduction for CPU savings
            if current_time - last_check_time < 1.5:  # Check only every 1.5 seconds
                time.sleep(0.8)  # Even longer sleep to maximize CPU savings
                continue
                
            last_check_time = current_time
            
            # Check if listening was just re-enabled by API
            if recognition_paused_for_listen_status and SHOULD_LISTEN.is_set():
                print("Listening enabled by API, restarting speech recognition")
                recognition_paused_for_listen_status = False
                # Reset consecutive stops counter since this is an intentional restart
                consecutive_recognition_stops = 0
                # Small delay to make sure everything is ready
                time.sleep(1.0)
                
            # Check if user is absent
            if USER_ABSENT.is_set():
                # If we haven't already paused recognition for absence
                if not recognition_paused_for_absence and recognition is not None:
                    print("User absent or too far, pausing speech recognition")
                    try:
                        recognition.stop()
                        print("Recognition service paused due to user absence")
                    except Exception as e:
                        print(f"Error pausing recognition: {e}")
                    recognition = None
                    recognition_paused_for_absence = True
                
                # Wait longer to reduce CPU usage during user absence
                time.sleep(3.0)  # Increased from 2.0 to 3.0 seconds for maximum CPU savings
                continue
            
            # User is present now, check if we need to restart recognition
            if recognition_paused_for_absence and recognition is None and SHOULD_LISTEN.is_set():
                print("User returned, restarting speech recognition")
                recognition_paused_for_absence = False
                # Reset consecutive stops counter since this is an intentional restart
                consecutive_recognition_stops = 0
                # Small delay to make sure everything is ready
                time.sleep(1.0)
            
            # Check if system is speaking
            if NOW_SPEAKING.locked():
                # If we haven't already paused recognition for speech
                if not recognition_paused_for_speech and recognition is not None:
                    print("System is speaking, pausing speech recognition")
                    try:
                        recognition.stop()
                        print("Recognition service paused")
                    except Exception as e:
                        print(f"Error pausing recognition: {e}")
                    recognition = None
                    recognition_paused_for_speech = True
                
                # Wait longer to reduce CPU usage during system speech
                time.sleep(1.5)  # Increased from 1.0 to 1.5 seconds for better CPU efficiency
                continue
            
            # System is not speaking now, check if we need to restart recognition
            if recognition_paused_for_speech and recognition is None and SHOULD_LISTEN.is_set():
                print("System speech ended, restarting speech recognition")
                recognition_paused_for_speech = False
                # Reset consecutive stops counter since this is an intentional restart
                consecutive_recognition_stops = 0
                # Small delay to make sure speech is completely finished
                time.sleep(1.0)
            
            # Check if we've exceeded max retries
            if retry_count >= MAX_RETRY_ATTEMPTS:
                print(f"Exceeded maximum retry attempts ({MAX_RETRY_ATTEMPTS}). Resetting retry count.")
                retry_count = 0
                time.sleep(5)  # Give a longer pause before starting fresh
                
                # Complete audio system reset after max retries
                try:
                    if stream is not None:
                        try:
                            stream.stop_stream()
                            stream.close()
                        except:
                            pass
                        stream = None
                        
                    if mic is not None:
                        try:
                            mic.terminate()
                        except:
                            pass
                        mic = None
                    print("Audio system reset after max retries")
                except Exception as reset_error:
                    print(f"Error during audio reset: {reset_error}")
            
            # Initialize recognition if needed (only if listening is enabled)
            if recognition is None and SHOULD_LISTEN.is_set():
                # Call recognition service by async mode
                recognition = Recognition(
                    model='paraformer-realtime-v2',
                    format=format_pcm,
                    sample_rate=sample_rate,
                    semantic_punctuation_enabled=False,
                    callback=callback)
                
                # Start translation
                recognition.start()
                print("Speech recognition started successfully")
                # Reset retry count on successful connection
                retry_count = 0
                last_activity_time = time.time()
            elif recognition is None and not SHOULD_LISTEN.is_set():
                # Skip initialization when listening is disabled
                time.sleep(2.0)
                continue
            
            # Only set up signal handler in the main thread
            is_main_thread = threading.current_thread() is threading.main_thread()
            if is_main_thread:
                signal.signal(signal.SIGINT, signal_handler)
                print("Press 'Ctrl+C' to stop recording and translation...")
            
            # Process audio frames
            connection_health_timer = time.time()
            while True:
                # Exit conditions that would require us to restart recognition
                if USER_ABSENT.is_set() or NOW_SPEAKING.locked():
                    # Need to break out and restart the main loop checks
                    break
                    
                current_time = time.time()
                
                # Check for recognition timeouts
                if current_time - last_activity_time > RECOGNITION_TIMEOUT:
                    print(f"Recognition timeout detected - no activity for {RECOGNITION_TIMEOUT} seconds")
                    # Force reconnection
                    raise TimeoutError("Recognition timeout")
                
                # Periodically check and refresh connection health (reduced frequency)
                if current_time - connection_health_timer > 900:  # Increased from 10 to 15 minutes for CPU savings
                    print("Performing periodic connection health check")
                    # Try sending a minimal audio frame to check connection
                    try:
                        recognition.send_audio_frame(b'\x00' * 32)  # Send minimal data
                        connection_health_timer = current_time
                    except Exception as e:
                        print(f"Connection health check failed: {e}")
                        # Break inner loop to trigger reconnection
                        break
                
                if stream and hasattr(stream, 'is_active') and stream.is_active():
                    # Skip processing if system state has changed
                    if USER_ABSENT.is_set() or NOW_SPEAKING.locked():
                        break
                    
                    try:
                        # Check stream is still active right before reading
                        if not stream.is_active():
                            print("Stream became inactive, recreating...")
                            raise IOError("Stream not active")
                            
                        data = stream.read(block_size, exception_on_overflow=False)
                        # Check data validity
                        expected_size = block_size * 2  # block_size frames * 2 bytes per 16-bit sample
                        if len(data) != expected_size:
                            print(f"Invalid data size: {len(data)}, expected: {expected_size}")
                            continue
                            
                        recognition.send_audio_frame(data)
                        last_activity_time = time.time()  # Update activity timestamp
                        
                        # Add small sleep to prevent excessive CPU usage in audio processing loop
                        time.sleep(0.01)  # 10ms sleep to give CPU breathing room
                    except (IOError, OSError) as e:
                        print(f"Audio I/O error: {e}")
                        # Try to reset the audio system
                        if callback:
                            callback._reset_audio()
                        break
                    except ValueError as e:
                        print(f"Value error: {e}")
                        # Likely a format issue with the audio stream
                        if stream is not None:
                            try:
                                stream.stop_stream()
                                stream.close()
                            except:
                                pass
                            stream = None
                        break
                    except dashscope.common.error.InvalidParameter as e:
                        print(f"DashScope error: {e}")
                        # Check if this is the "Speech recognition has stopped" error
                        if "Speech recognition has stopped" in str(e):
                            consecutive_recognition_stops += 1
                            print(f"Recognition stopped error #{consecutive_recognition_stops}")
                            
                            # After multiple consecutive failures, implement more aggressive recovery
                            if consecutive_recognition_stops >= 3:
                                print("Multiple recognition failures detected. Performing full system reset...")
                                # Reset everything
                                if recognition is not None:
                                    try:
                                        recognition.stop()
                                    except:
                                        pass
                                    recognition = None
                                
                                if stream is not None:
                                    try:
                                        stream.stop_stream()
                                        stream.close()
                                    except:
                                        pass
                                    stream = None
                                
                                if mic is not None:
                                    try:
                                        mic.terminate()
                                    except:
                                        pass
                                    mic = None
                                
                                # Longer backoff with full reset
                                wait_time = 5 + consecutive_recognition_stops
                                print(f"Waiting {wait_time} seconds before full restart...")
                                time.sleep(wait_time)
                                consecutive_recognition_stops = 0  # Reset counter after recovery
                            else:
                                # Regular recovery for first few failures
                                # Force a complete restart of the recognition service
                                if recognition is not None:
                                    try:
                                        recognition.stop()
                                    except Exception as cleanup_error:
                                        print(f"Error during recognition cleanup: {cleanup_error}")
                                    recognition = None
                                
                                print(f"Restarting speech recognition service (attempt {consecutive_recognition_stops})...")
                                # Small delay to allow for clean reconnection
                                time.sleep(consecutive_recognition_stops * 2)  # Increasing delay with each failure
                        else:
                            # Different InvalidParameter error
                            if recognition is not None:
                                try:
                                    recognition.stop()
                                except Exception as cleanup_error:
                                    print(f"Error during recognition cleanup: {cleanup_error}")
                                recognition = None
                            
                            print("Restarting speech recognition service due to parameter error...")
                            time.sleep(2)
                            consecutive_recognition_stops = 0  # Reset counter for different errors
                        
                        # Break out of the inner loop to trigger reconnection
                        break
                    except aiohttp.client_exceptions.ClientConnectionResetError as e:
                        print(f"WebSocket connection reset: {e}")
                        time.sleep(1)  # Brief pause before reconnecting
                        break
                    except Exception as e:
                        print(f"Error processing audio frame: {e}")
                        # Break the inner loop to reinitialize recognition
                        break
                else:
                    # Stream not active or doesn't exist
                    print("Stream not active, recreating...")
                    # Try to reset the audio
                    if callback:
                        callback._reset_audio()
                    time.sleep(0.5)
                    break
                    
        except TimeoutError as e:
            print(f"Timeout error: {e}")
            retry_count += 1
            # Clean up and reconnect
            if recognition is not None:
                try:
                    recognition.stop()
                except Exception as cleanup_error:
                    print(f"Error stopping recognition during timeout: {cleanup_error}")
                recognition = None
            time.sleep(2)  # Brief pause before reconnecting
            
        except OSError as e:
            print(f"OS error in mic_listen: {e}")
            # OS errors need special handling - complete reset
            retry_count += 1
            
            # Clean up everything
            if recognition is not None:
                try:
                    recognition.stop()
                except:
                    pass
                recognition = None
                
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
                stream = None
                
            if mic is not None:
                try:
                    mic.terminate()
                except:
                    pass
                mic = None
                
            # Wait longer for OS errors
            wait_time = 5 + retry_count * 2
            print(f"OS error recovery - waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
        except aiohttp.client_exceptions.ClientConnectionResetError as e:
            print(f"WebSocket connection reset: {e}")
            retry_count += 1
            # For connection resets, wait a bit longer to allow server to recover
            if recognition is not None:
                try:
                    recognition.stop()
                except:
                    pass
                recognition = None
            time.sleep(5)
            
        except Exception as e:
            print(f"Error in mic_listen: {e}")
            # Implement exponential backoff with jitter for reconnection
            backoff_time = INITIAL_BACKOFF * (2 ** min(retry_count, 6))  # Cap at 2^6
            jitter = random.uniform(0, 0.5 * backoff_time)  # Add up to 50% jitter
            retry_wait = backoff_time + jitter
            retry_count += 1
            
            # Clean up resources
            if recognition is not None:
                try:
                    recognition.stop()
                except Exception as cleanup_error:
                    print(f"Error stopping recognition: {cleanup_error}")
                recognition = None
            
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as cleanup_error:
                    print(f"Error closing stream: {cleanup_error}")
                stream = None
                
            if mic is not None:
                try:
                    mic.terminate()
                except Exception as cleanup_error:
                    print(f"Error terminating mic: {cleanup_error}")
                mic = None
                
            # Wait before retrying with backoff
            print(f"Attempt {retry_count}/{MAX_RETRY_ATTEMPTS}: Will attempt to reconnect in {retry_wait:.2f} seconds...")
            time.sleep(retry_wait)

            # When we reconnect successfully, reset the consecutive stops counter
            if recognition is not None and consecutive_recognition_stops > 0:
                consecutive_recognition_stops = 0
                print("Successfully reconnected, resetting failure counters")

# main function
if __name__ == '__main__':
    init_dashscope_api_key()
    print('Initializing ...')

    # Create the translation callback
    # callback = Callback()

    # # Call recognition service by async mode, you can customize the recognition parameters, like model, format,
    # # sample_rate For more information, please refer to https://help.aliyun.com/document_detail/2712536.html
    # recognition = Recognition(
    #     model='paraformer-realtime-v2',
    #     # 'paraformer-realtime-v1'、'paraformer-realtime-8k-v1'
    #     format=format_pcm,
    #     # 'pcm'、'wav'、'opus'、'speex'、'aac'、'amr', you can check the supported formats in the document
    #     sample_rate=sample_rate,
    #     # support 8000, 16000
    #     semantic_punctuation_enabled=False,
    #     callback=callback)

    # # Start translation
    # recognition.start()

    # signal.signal(signal.SIGINT, signal_handler)
    # print("Press 'Ctrl+C' to stop recording and translation...")
    # # Create a keyboard listener until "Ctrl+C" is pressed

    # while True:
    #     if stream:
    #         data = stream.read(3200, exception_on_overflow=False)
    #         recognition.send_audio_frame(data)
    #     else:
    #         break

    # recognition.stop()
