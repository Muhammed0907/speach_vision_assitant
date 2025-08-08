#!/usr/bin/env python3
import subprocess

def check_audio_status():
    """Check current audio status and suggest fixes"""
    print("=== Audio System Status ===\n")
    
    try:
        # Check sinks
        print("1. Available sinks:")
        result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        
        # Check sink volume details
        print("2. Sink volume details:")
        result = subprocess.run(['pactl', 'list', 'sinks'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'Sink #' in line or 'Volume:' in line or 'Mute:' in line or 'Name:' in line:
                    print(line.strip())
        
        # Check default sink
        print("\n3. Default sink:")
        result = subprocess.run(['pactl', 'get-default-sink'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            default_sink = result.stdout.strip()
            print(f"Default: {default_sink}")
            
            # Get volume of default sink
            result = subprocess.run(['pactl', 'get-sink-volume', default_sink], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Volume: {result.stdout.strip()}")
        
        # Check if any applications are playing audio
        print("\n4. Audio applications:")
        result = subprocess.run(['pactl', 'list', 'sink-inputs', 'short'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print(result.stdout)
            else:
                print("No applications currently playing audio")
                
    except Exception as e:
        print(f"Error checking audio status: {e}")

if __name__ == "__main__":
    check_audio_status()