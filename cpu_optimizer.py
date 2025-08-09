#!/usr/bin/env python3
"""
CPU Optimization Utility for AI Kiosk Application
Provides real-time CPU monitoring and adaptive performance adjustments
"""

import threading
import time
import os
import sys
from typing import Dict, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not available. Install with: uv add psutil")
    PSUTIL_AVAILABLE = False



class CPUOptimizer:
    """
    Real-time CPU optimization manager for the AI kiosk application.
    Monitors CPU usage and provides adaptive performance adjustments.
    """
    
    def __init__(self, target_cpu_usage: float = 70.0):
        self.target_cpu_usage = target_cpu_usage
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance adjustment settings
        self.performance_levels = {
            'ultra_high': {
                'face_detection_fps': 5,
                'face_detection_interval': 0.3,
                'audio_block_size': 6400,
                'detection_size': (736, 736),
                'camera_resolution': (480, 360),
                'sleep_multiplier': 1.0
            },
            'high': {
                'face_detection_fps': 4,
                'face_detection_interval': 0.4,
                'audio_block_size': 8000,
                'detection_size': (736, 736),
                'camera_resolution': (440, 330),
                'sleep_multiplier': 1.2
            },
            'medium': {
                'face_detection_fps': 3,
                'face_detection_interval': 0.6,
                'audio_block_size': 9600,
                'detection_size': (736, 736),
                'camera_resolution': (400, 300),
                'sleep_multiplier': 1.5
            },
            'low': {
                'face_detection_fps': 2,
                'face_detection_interval': 0.8,
                'audio_block_size': 12800,
                'detection_size': (736, 736),
                'camera_resolution': (360, 270),
                'sleep_multiplier': 2.0
            },
            'ultra_low': {
                'face_detection_fps': 1,
                'face_detection_interval': 1.0,
                'audio_block_size': 16000,
                'detection_size': (736, 736),
                'camera_resolution': (320, 240),
                'sleep_multiplier': 3.0
            }
        }
        
        self.current_level = 'medium'  # Start with medium performance
        self.cpu_history = []
        self.adjustment_cooldown = 5.0  # Seconds between adjustments
        self.last_adjustment = 0
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not PSUTIL_AVAILABLE:
            return 50.0  # Default assumption
        
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            print(f"Error getting CPU usage: {e}")
            return 50.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not PSUTIL_AVAILABLE:
            return {'percent': 50.0, 'available': 1000}
        
        try:
            memory = psutil.virtual_memory()
            return {
                'percent': memory.percent,
                'available': memory.available / (1024 * 1024)  # MB
            }
        except Exception as e:
            print(f"Error getting memory usage: {e}")
            return {'percent': 50.0, 'available': 1000}
    
    def analyze_performance_need(self) -> str:
        """
        Analyze current system performance and determine optimal level.
        Returns the recommended performance level.
        """
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        
        # Add to history for trend analysis
        self.cpu_history.append(cpu_usage)
        if len(self.cpu_history) > 10:
            self.cpu_history.pop(0)
        
        # Calculate average CPU usage over recent history
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        
        # Determine appropriate performance level
        if avg_cpu > 85 or memory_usage['percent'] > 90:
            return 'ultra_low'
        elif avg_cpu > 75 or memory_usage['percent'] > 80:
            return 'low'
        elif avg_cpu > 65 or memory_usage['percent'] > 70:
            return 'medium'
        elif avg_cpu > 50:
            return 'high'
        else:
            return 'ultra_high'
    
    def get_performance_settings(self, level: str = None) -> Dict[str, Any]:
        """Get performance settings for specified level."""
        if level is None:
            level = self.current_level
        
        return self.performance_levels.get(level, self.performance_levels['medium'])
    
    def adjust_performance_level(self, new_level: str) -> bool:
        """
        Adjust performance level if needed and enough time has passed.
        Returns True if adjustment was made.
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return False
        
        # Don't adjust if already at the right level
        if new_level == self.current_level:
            return False
            
        old_level = self.current_level
        self.current_level = new_level
        self.last_adjustment = current_time
        
        print(f"CPU Optimizer: Performance level changed from {old_level} to {new_level}")
        print(f"  New settings: {self.get_performance_settings()}")
        
        return True
    
    def monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        print("CPU Optimizer: Monitoring started")
        
        while self.monitoring_active:
            try:
                # Analyze current performance needs
                recommended_level = self.analyze_performance_need()
                
                # Adjust if needed
                if self.adjust_performance_level(recommended_level):
                    # Log current system stats
                    cpu_usage = self.get_cpu_usage()
                    memory_usage = self.get_memory_usage()
                    print(f"CPU Optimizer: Current stats - CPU: {cpu_usage:.1f}%, Memory: {memory_usage['percent']:.1f}%")
                
                # Wait before next check
                time.sleep(2.0)
                
            except Exception as e:
                print(f"CPU Optimizer error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def start_monitoring(self):
        """Start the CPU monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("CPU Optimizer: Monitoring thread started")
    
    def stop_monitoring(self):
        """Stop the CPU monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        print("CPU Optimizer: Monitoring stopped")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage': self.get_memory_usage(),
            'current_performance_level': self.current_level,
            'performance_settings': self.get_performance_settings()
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info.update({
                    'cpu_count': psutil.cpu_count(),
                    'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                })
            except Exception as e:
                print(f"Error getting extended system info: {e}")
        
        return info
    
    def print_system_stats(self):
        """Print current system statistics."""
        info = self.get_system_info()
        print("\n=== CPU Optimizer System Stats ===")
        print(f"CPU Usage: {info['cpu_usage']:.1f}%")
        print(f"Memory Usage: {info['memory_usage']['percent']:.1f}% ({info['memory_usage']['available']:.0f}MB available)")
        print(f"Performance Level: {info['current_performance_level']}")
        print(f"Face Detection FPS: {info['performance_settings']['face_detection_fps']}")
        print(f"Detection Size: {info['performance_settings']['detection_size']}")
        print(f"Audio Block Size: {info['performance_settings']['audio_block_size']}")
        if 'cpu_count' in info:
            print(f"CPU Cores: {info['cpu_count']}")
        print("=" * 35)

# Global optimizer instance
cpu_optimizer = CPUOptimizer()

def get_optimizer() -> CPUOptimizer:
    """Get the global CPU optimizer instance."""
    return cpu_optimizer

def optimize_process_priority():
    """Set process priority for better CPU scheduling."""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        # Lower the process priority to be more cooperative
        process = psutil.Process()
        if sys.platform == 'win32':
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(5)  # Positive nice value = lower priority
        print("CPU Optimizer: Process priority optimized")
    except Exception as e:
        print(f"CPU Optimizer: Could not adjust process priority: {e}")

def enable_cpu_affinity_optimization():
    """Optimize CPU affinity if possible."""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        process = psutil.Process()
        cpu_count = psutil.cpu_count()
        
        if cpu_count > 4:
            # Use only some cores to leave others free for system
            cores_to_use = list(range(cpu_count // 2))
            process.cpu_affinity(cores_to_use)
            print(f"CPU Optimizer: Limited to cores {cores_to_use}")
    except Exception as e:
        print(f"CPU Optimizer: Could not optimize CPU affinity: {e}")

if __name__ == "__main__":
    # Test the optimizer
    optimizer = get_optimizer()
    
    # Apply system optimizations
    optimize_process_priority() 
    enable_cpu_affinity_optimization()
    
    # Start monitoring
    optimizer.start_monitoring()
    
    try:
        while True:
            optimizer.print_system_stats()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping CPU optimizer...")
        optimizer.stop_monitoring()