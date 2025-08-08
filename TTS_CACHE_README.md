# TTS Caching System

This document explains the TTS (Text-to-Speech) caching system implemented for the emotion detection project to optimize performance and reduce API calls to Qwen's Dashscope service.

## Overview

The TTS caching system stores generated audio files locally to avoid regenerating the same text multiple times. This is particularly useful for:

- **Greetings**: Frequently repeated welcome messages
- **Suggestions**: Product recommendations that are reused
- **No-Person Suggestions**: Messages when no one is detected
- **Busy Speak**: Charging/maintenance announcements

## How It Works

### 1. Cache Structure
```
tts_cache/
├── cache_index.json          # Index of cached entries with metadata
├── a1b2c3d4e5f6.mp3         # Cached audio file (MD5 hash)
├── f6e5d4c3b2a1.mp3         # Another cached audio file
└── ...
```

### 2. Cache Process
1. **Check Cache**: Before calling Qwen API, check if the text has been cached
2. **Play Cached**: If found, play the cached audio file directly
3. **Generate & Cache**: If not found, generate TTS via API and save to cache
4. **Index Update**: Update the cache index with metadata

### 3. Cache Key Generation
Cache keys are generated using MD5 hash of: `{text}_{voice}_{model}`

Example: `"欢迎光临_longke_cosyvoice-v1"` → `a1b2c3d4e5f6789...`

## Benefits

### Performance Improvements
- **Instant Playback**: Cached audio plays immediately (no API delay)
- **Reduced Latency**: ~50-200ms vs 1-3 seconds for API calls
- **Bandwidth Savings**: No repeated network requests for same content

### Cost Optimization
- **API Call Reduction**: Dramatically reduces Qwen API usage
- **Resource Efficiency**: Less CPU/memory usage for repeated content

### User Experience
- **Faster Response**: Immediate greetings and suggestions
- **Consistent Quality**: Same audio quality for repeated phrases
- **Offline Capability**: Works without internet for cached content

## Usage

### Basic TTS (Automatic Caching)
```python
from qwenai import synthesis_text_to_speech_and_play_by_streaming_mode

# This will check cache first, then generate if needed
synthesis_text_to_speech_and_play_by_streaming_mode("欢迎光临")
```

### Cache Management
```python
from qwenai import get_tts_cache_stats, clear_tts_cache

# Get cache statistics
stats = get_tts_cache_stats()
print(f"Cache has {stats['valid_entries']} entries, {stats['total_size_mb']} MB")

# Clear old cache entries (older than 7 days)
clear_tts_cache(older_than_days=7)
```

### Using the Cache Manager Script
```bash
# Show cache statistics
python tts_cache_manager.py stats

# Clear all cache
python tts_cache_manager.py clear

# Clear entries older than 7 days
python tts_cache_manager.py clear --days 7

# Test caching performance
python tts_cache_manager.py test
```

## Cache Configuration

The cache system automatically:
- Creates `tts_cache/` directory in the project root
- Manages cache index in `cache_index.json`
- Validates cached files on access
- Removes invalid entries automatically

## Expected Cache Usage

For a typical deployment with default messages:

### Greetings (4-8 entries)
- "嗨~你好呀！欢迎来到..." 
- "哇~新朋友来啦..."
- Gender-specific variations

### Suggestions (10-20 entries)
- "忙碌一天啦，下班路上来抽个盲盒..."
- "给身边的小伙伴抽个盲盒..."
- Various product recommendations

### System Messages (5-10 entries)
- "在充电。"
- "本宝宝正在充电中..."
- Status announcements

### Total Expected Cache Size
- ~50-100 entries
- ~5-20 MB disk space
- 90%+ cache hit rate for repeated content

## Implementation Details

### Cache Validation
- Checks file existence and size > 0
- Removes invalid entries automatically
- Rebuilds index if corrupted

### Thread Safety
- Cache operations are thread-safe
- Multiple TTS requests handled correctly
- No race conditions in cache updates

### Error Handling
- Graceful fallback to API if cache fails
- Automatic retry for corrupted files
- Logging of cache operations

## Monitoring

Monitor cache effectiveness:
```python
stats = get_tts_cache_stats()
hit_rate = (stats['valid_entries'] / total_tts_calls) * 100
print(f"Cache hit rate: {hit_rate:.1f}%")
```

## Maintenance

### Automatic Maintenance
- Invalid entries removed on access
- Index updated after each operation
- Self-healing cache system

### Manual Maintenance
```bash
# Weekly cleanup (recommended)
python tts_cache_manager.py clear --days 7

# Monthly full cleanup
python tts_cache_manager.py clear

# Monitor cache size
python tts_cache_manager.py stats
```

## Project Integration

The caching system is integrated into:

1. **main.py**: Automatic caching for all TTS calls
2. **qwenai.py**: Core caching implementation
3. **Queue speech functions**: Cache-aware speech queuing

### Key Functions Using Cache
- `queue_speech()` in main.py
- `synthesis_text_to_speech_and_play_by_streaming_mode()` 
- All greeting, suggestion, and announcement functions

This TTS caching system provides significant performance improvements while maintaining the same API compatibility as the original implementation.