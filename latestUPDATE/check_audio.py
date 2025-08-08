#!/usr/bin/env python3
import pyaudio
import sys

def check_device_sample_rates():
    pa = pyaudio.PyAudio()
    
    print("Available audio devices and supported sample rates:")
    print("=" * 60)
    
    common_rates = [8000, 11025, 16000, 22050, 44100, 48000, 96000]
    
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"\nDevice {i}: {info['name']}")
        print(f"  Max input channels: {info['maxInputChannels']}")
        print(f"  Max output channels: {info['maxOutputChannels']}")
        print(f"  Default sample rate: {info['defaultSampleRate']}")
        
        if info['maxInputChannels'] > 0:
            print("  Supported input sample rates:", end=" ")
            supported_rates = []
            for rate in common_rates:
                try:
                    if pa.is_format_supported(
                        rate=rate,
                        input_device=i,
                        input_channels=1,
                        input_format=pyaudio.paInt16
                    ):
                        supported_rates.append(str(rate))
                except ValueError:
                    pass
            print(", ".join(supported_rates) if supported_rates else "None")
            
        if info['maxOutputChannels'] > 0:
            print("  Supported output sample rates:", end=" ")
            supported_rates = []
            for rate in common_rates:
                try:
                    if pa.is_format_supported(
                        rate=rate,
                        output_device=i,
                        output_channels=1,
                        output_format=pyaudio.paInt16
                    ):
                        supported_rates.append(str(rate))
                except ValueError:
                    pass
            print(", ".join(supported_rates) if supported_rates else "None")
    
    pa.terminate()

if __name__ == "__main__":
    check_device_sample_rates()