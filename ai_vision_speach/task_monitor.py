import requests
import time
import threading
import json
from typing import Dict, Any, Optional

class TaskMonitor:
    """Monitor task status from API and control application execution"""
    
    def __init__(self, machine_id: str = "6", api_url: str = "https://manghe.shundaocehua.cn/screen/task-data/detail"):
        self.machine_id = machine_id
        self.api_url = api_url
        self.full_url = f"{api_url}/{machine_id}"
        self.monitor_interval = 60  # 1 minute
        self.is_running = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Application control flags
        self.should_run_application = True
        self.last_task_data = None
        
        # Callbacks for application control
        self.on_application_start = None
        self.on_application_stop = None
        
    def fetch_task_data(self) -> Optional[Dict[str, Any]]:
        """Fetch task data from API"""
        try:
            response = requests.get(self.full_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == 0:
                return data.get("data")
            else:
                print(f"API returned error code: {data.get('code')}, message: {data.get('msg')}")
                return None
                
        except requests.RequestException as e:
            print(f"Failed to fetch task data: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse API response: {e}")
            return None
    
    def should_application_run(self, task_data: Dict[str, Any]) -> bool:
        """
        Determine if application should run based on task data
        Application should run if:
        - taskStatus is "正在执行" OR taskBusy is True
        """
        if not task_data:
            return False
            
        task_status = task_data.get("taskStatus", "")
        task_busy = task_data.get("taskBusy", False)
        
        # Application should run if task is executing or busy
        should_run = task_status == "正在执行" and task_busy
        # should_run = False
        
        print(f"Task Status: {task_status}, Task Busy: {task_busy}, Should Run: {should_run}")
        return should_run
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                task_data = self.fetch_task_data()
                
                if task_data:
                    self.last_task_data = task_data
                    
                    # Print task information
                    print(f"Task Monitor - Machine ID: {task_data.get('machineId')}")
                    print(f"Task Name: {task_data.get('taskName')}")
                    print(f"Current Task: {task_data.get('currentTaskName')}")
                    print(f"Task Status: {task_data.get('taskStatus')}")
                    print(f"Task Busy: {task_data.get('taskBusy')}")
                    print(f"Battery: {task_data.get('batteryPercent')}%")
                    print(f"Data Time: {task_data.get('dataTime')}")
                    
                    # Check if application should run
                    should_run = self.should_application_run(task_data)
                    
                    # Update application state if changed
                    if should_run != self.should_run_application:
                        self.should_run_application = should_run
                        
                        if should_run:
                            print("=== APPLICATION STARTED ===")
                            if self.on_application_start:
                                self.on_application_start(task_data)
                        else:
                            print("=== APPLICATION STOPPED ===")
                            if self.on_application_stop:
                                self.on_application_stop(task_data)
                    
                    print("-" * 50)
                else:
                    print("Failed to fetch task data, keeping current state")
                    
            except Exception as e:
                print(f"Task Monitor Error: {e}")
            
            # Wait for next check
            time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Task monitoring started for machine ID: {self.machine_id}")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            
        print("Task monitoring stopped")
    
    def get_last_task_data(self) -> Optional[Dict[str, Any]]:
        """Get the last fetched task data"""
        return self.last_task_data
    
    def is_application_running(self) -> bool:
        """Check if application should be running"""
        return self.should_run_application
    
    def set_application_callbacks(self, on_start=None, on_stop=None):
        """Set callbacks for application start/stop events"""
        self.on_application_start = on_start
        self.on_application_stop = on_stop