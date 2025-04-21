import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", algorithm_name="simulation"):
        # Create main logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create subfolder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_subfolder = os.path.join(log_dir, timestamp)
        if not os.path.exists(self.log_subfolder):
            os.makedirs(self.log_subfolder)
            
        # Create log file with algorithm name
        self.log_file = os.path.join(self.log_subfolder, f"{algorithm_name}.txt")
        
        # Open file in append mode
        self.file = open(self.log_file, 'a')
        
    def log(self, message: str, prefix: str = ""):
        """Log a message with optional prefix"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {prefix}{message}\n"
        self.file.write(log_message)
        self.file.flush()  # Ensure message is written immediately
        
    def close(self):
        """Close the log file"""
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
        
    def __del__(self):
        """Ensure file is closed when object is destroyed"""
        if hasattr(self, 'file') and not self.file.closed:
            self.close()
 