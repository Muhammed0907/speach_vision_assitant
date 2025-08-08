#!/usr/bin/env python3
import subprocess
import sys

def test_volume_control(volume_percent=60):
    """Test volume control with debugging"""
    print(f"Testing volume control with {volume_percent}%...")
    
    # Convert to decimal
    pactl_volume = volume_percent / 100.0
    
    try:
        # Get sinks
        print("Getting PulseAudio/PipeWire sinks...")
        result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("Available sinks:")
            print(result.stdout)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    sink_id = parts[0]
                    sink_name = parts[1]
                    
                    print(f"\nSetting sink {sink_id} ({sink_name}) to {volume_percent}%...")
                    
                    # Unmute first
                    subprocess.run(['pactl', 'set-sink-mute', sink_id, '0'], timeout=3)
                    print(f"Unmuted sink {sink_id}")
                    
                    # Set volume
                    subprocess.run(['pactl', 'set-sink-volume', sink_id, str(pactl_volume)], timeout=3)
                    print(f"Set volume to {pactl_volume}")
                    
                    # Get current volume to verify
                    vol_result = subprocess.run(['pactl', 'get-sink-volume', sink_id], 
                                              capture_output=True, text=True, timeout=3)
                    if vol_result.returncode == 0:
                        print(f"Current volume: {vol_result.stdout.strip()}")
                    
                    # Also try percentage format
                    subprocess.run(['pactl', 'set-sink-volume', sink_id, f'{volume_percent}%'], timeout=3)
                    print(f"Also tried percentage format: {volume_percent}%")
        else:
            print(f"Failed to get sinks: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    vol = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    test_volume_control(vol)