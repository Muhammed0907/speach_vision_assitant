#!/usr/bin/env python3
"""
TTS Cache Manager

A utility script to manage the TTS cache for the emotion detection project.
Provides functions to view cache statistics, clear old entries, and maintain the cache.

Usage:
    python tts_cache_manager.py stats              # Show cache statistics
    python tts_cache_manager.py clear              # Clear all cache
    python tts_cache_manager.py clear --days 7     # Clear entries older than 7 days
    python tts_cache_manager.py test               # Test TTS with caching
"""

import argparse
import sys
import time
from qwenai import get_tts_cache_stats, clear_tts_cache, synthesis_text_to_speech_and_play_by_streaming_mode

def show_stats():
    """Show TTS cache statistics"""
    stats = get_tts_cache_stats()
    print("TTS Cache Statistics:")
    print("=" * 40)
    print(f"Cache Directory: {stats['cache_dir']}")
    print(f"Total Entries: {stats['total_entries']}")
    print(f"Valid Entries: {stats['valid_entries']}")
    print(f"Invalid Entries: {stats['invalid_entries']}")
    print(f"Total Size: {stats['total_size_mb']} MB")
    
    if stats['total_entries'] > 0:
        cache_efficiency = (stats['valid_entries'] / stats['total_entries']) * 100
        print(f"Cache Efficiency: {cache_efficiency:.1f}%")

def clear_cache(days=None):
    """Clear cache entries"""
    if days:
        print(f"Clearing cache entries older than {days} days...")
        cleared = clear_tts_cache(older_than_days=days)
        print(f"Cleared {cleared} entries older than {days} days")
    else:
        print("Clearing all cache entries...")
        cleared = clear_tts_cache()
        print(f"Cleared {cleared} entries")

def test_tts():
    """Test TTS with caching"""
    test_texts = [
        "欢迎光临",
        "您好，需要推荐吗？",
        "今天天气真不错",
        "感谢您的光临"
    ]
    
    print("Testing TTS caching...")
    print("=" * 40)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        print("First call (should generate TTS):")
        start_time = time.time()
        synthesis_text_to_speech_and_play_by_streaming_mode(text)
        first_duration = time.time() - start_time
        print(f"Duration: {first_duration:.2f}s")
        
        # Small delay between calls
        time.sleep(1)
        
        print("Second call (should use cache):")
        start_time = time.time()
        synthesis_text_to_speech_and_play_by_streaming_mode(text)
        second_duration = time.time() - start_time
        print(f"Duration: {second_duration:.2f}s")
        
        speedup = first_duration / second_duration if second_duration > 0 else 0
        print(f"Speedup: {speedup:.1f}x")
        
        time.sleep(2)  # Delay between different texts

def main():
    parser = argparse.ArgumentParser(description='TTS Cache Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    subparsers.add_parser('stats', help='Show cache statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache entries')
    clear_parser.add_argument('--days', type=int, help='Clear entries older than N days')
    
    # Test command
    subparsers.add_parser('test', help='Test TTS caching performance')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'stats':
            show_stats()
        elif args.command == 'clear':
            clear_cache(args.days)
        elif args.command == 'test':
            test_tts()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()