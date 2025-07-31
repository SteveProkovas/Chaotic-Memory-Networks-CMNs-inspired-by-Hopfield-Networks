import psutil
import numpy as np
import os
import gc
import time
from threading import Thread, Event
import logging

class HardwareResourceMonitor:
    """Continuous hardware monitor for memory-constrained systems"""
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.stop_event = Event()
        self.monitor_thread = None
        
        # Hardware specs
        self.total_memory = psutil.virtual_memory().total
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        
        # Current state
        self.memory_usage = 0.0
        self.cpu_usage = 0.0
        self.swap_usage = 0.0
        self.temperature = 0.0
        self.safe_threshold = 0.85  # 85% resource usage
        
        # Initialize logging
        logging.basicConfig(
            filename='memory_manager.log', 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Hardware Monitor Initialized | "
                     f"RAM: {self.total_memory/1e9:.2f}GB | "
                     f"Cores: {self.cpu_cores}/{self.cpu_threads}")
    
    def start(self):
        """Begin background monitoring"""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logging.info("Resource monitoring started")
    
    def stop(self):
        """Stop background monitoring"""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logging.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring thread"""
        while not self.stop_event.is_set():
            try:
                # Memory metrics
                mem = psutil.virtual_memory()
                self.memory_usage = mem.percent / 100.0
                self.swap_usage = psutil.swap_memory().percent / 100.0
                
                # CPU metrics
                self.cpu_usage = psutil.cpu_percent() / 100.0
                
                # Temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if 'k10temp' in temps:  # AMD Ryzen sensors
                        self.temperature = temps['k10temp'][0].current
                except Exception as e:
                    logging.warning(f"Temperature reading failed: {str(e)}")
                
            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")
            
            time.sleep(self.update_interval)
    
    def current_status(self):
        """Get current resource status snapshot"""
        return {
            "memory_used": self.memory_usage,
            "cpu_used": self.cpu_usage,
            "swap_used": self.swap_usage,
            "temperature": self.temperature,
            "is_safe": self.is_operation_safe()
        }
    
    def is_operation_safe(self, additional_need=0):
        """
        Check if system has resources for new operations
        
        Args:
            additional_need: Estimated additional memory needed in bytes
        """
        available_mem = (1 - self.memory_usage) * self.total_memory
        if available_mem < additional_need:
            logging.warning(f"Insufficient memory: {available_mem/1e6:.2f}MB "
                           f"available < {additional_need/1e6:.2f}MB needed")
            return False
            
        if self.cpu_usage > self.safe_threshold:
            logging.warning(f"High CPU load: {self.cpu_usage*100:.1f}%")
            return False
            
        if self.swap_usage > 0.1:  # More than 10% swap usage
            logging.warning(f"Swap memory active: {self.swap_usage*100:.1f}%")
            return False
            
        if self.temperature > 85:  # Ryzen thermal throttling threshold
            logging.warning(f"High temperature: {self.temperature}°C")
            return False
            
        return True
    
    def estimate_available_memory(self):
        """Calculate available memory with safety margin"""
        safety_margin = 0.1  # Reserve 10% for system operations
        return (1 - self.memory_usage - safety_margin) * self.total_memory
    
    def force_memory_cleanup(self):
        """Aggressive memory cleanup when resources are low"""
        logging.info("Initiating forced memory cleanup")
        gc.collect()  # Explicit garbage collection
        time.sleep(0.5)  # Allow time for cleanup
        
        # Clear numpy internal cache
        try:
            np.savez_compressed('temp_cache_clear.npz', data=[])
            os.remove('temp_cache_compressed.npz')
        except:
            pass


class MemoryAwareExecutor:
    """Resource-constrained execution manager"""
    def __init__(self, resource_monitor):
        self.monitor = resource_monitor
        self.monitor.start()
        
    def __del__(self):
        self.monitor.stop()
    
    def auto_batch_size(self, pattern_size, n_patterns, dtype=np.float32):
        """
        Calculate safe batch size based on:
        - Pattern dimensionality
        - Number of patterns
        - Current system load
        - Data precision
        """
        # Calculate per-pattern memory requirement
        bytes_per_element = np.dtype(dtype).itemsize
        pattern_bytes = pattern_size * bytes_per_element
        
        # Memory needed for input + output + working memory
        # Working memory estimate: 2x input size for computations
        per_batch_mem = pattern_bytes * 3
        
        # Get available memory
        available_mem = self.monitor.estimate_available_memory()
        
        # Calculate max possible batches
        max_batches = max(1, int(available_mem / per_batch_mem))
        
        # Apply hardware constraints
        max_batches = min(max_batches, n_patterns, 100)  # Never exceed 100 batches
        
        logging.info(f"Batch calculation: {max_batches} batches | "
                    f"Pattern size: {pattern_size} | "
                    f"Available mem: {available_mem/1e6:.2f}MB")
        
        return max_batches
    
    def memory_safe_array(self, shape, dtype=np.float32, name="array"):
        """
        Create array with memory safety checks
        - Use memmap if too large for RAM
        """
        bytes_needed = np.prod(shape) * np.dtype(dtype).itemsize
        
        if not self.monitor.is_operation_safe(additional_need=bytes_needed):
            # Use disk-backed storage
            logging.warning(f"Creating disk-backed array for {name} "
                           f"({bytes_needed/1e6:.2f}MB)")
            return np.memmap(f'{name}.dat', dtype=dtype, 
                            mode='w+', shape=shape)
        
        # Create in-RAM array
        return np.zeros(shape, dtype=dtype)
    
    def execute_safely(self, operation, *args, **kwargs):
        """
        Execute operation with resource monitoring and fallbacks
        
        Args:
            operation: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            - fallback: Alternative operation if resources low
            - retries: Number of retry attempts (default 3)
        """
        retries = kwargs.pop('retries', 3)
        fallback = kwargs.pop('fallback', None)
        
        for attempt in range(retries):
            try:
                if not self.monitor.is_operation_safe():
                    if attempt == 0:
                        self.monitor.force_memory_cleanup()
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff
                
                return operation(*args, **kwargs)
            
            except MemoryError:
                logging.error(f"Memory error on attempt {attempt+1}/{retries}")
                self.monitor.force_memory_cleanup()
                if attempt == retries - 1 and fallback:
                    logging.warning("Using fallback operation")
                    return fallback(*args, **kwargs)
        
        raise MemoryError(f"Operation failed after {retries} attempts")
    
    def precision_optimizer(self, pattern_size, n_patterns):
        """Determine optimal precision based on problem scale"""
        # Calculate memory requirements for different precisions
        float32_req = pattern_size * n_patterns * 4
        float16_req = pattern_size * n_patterns * 2
        
        available_mem = self.monitor.estimate_available_memory()
        
        # Use float16 if float32 would exceed 60% of available memory
        if float32_req > available_mem * 0.6:
            logging.info(f"Using float16 precision: "
                        f"Float32 needs {float32_req/1e6:.2f}MB, "
                        f"Float16 needs {float16_req/1e6:.2f}MB")
            return np.float16
        
        return np.float32


# Example usage
if __name__ == "__main__":
    # Initialize monitoring system
    monitor = HardwareResourceMonitor()
    executor = MemoryAwareExecutor(monitor)
    
    try:
        # Create large array with safety checks
        large_array = executor.memory_safe_array(
            shape=(5000, 5000),  # 100MB at float32
            dtype=np.float32,
            name="experiment_data"
        )
        
        # Process in safe batches
        patterns = np.random.rand(100, 1000)
        batch_size = executor.auto_batch_size(
            pattern_size=1000,
            n_patterns=100,
            dtype=np.float32
        )
        
        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i+batch_size]
            # Process batch...
            print(f"Processing batch {i//batch_size + 1}")
            time.sleep(0.1)  # Simulate work
        
        # Check system status
        print("\nCurrent Resource Status:")
        for k, v in monitor.current_status().items():
            print(f"{k:>15}: {v}")
            
    finally:
        # Clean up memmap files
        if os.path.exists('experiment_data.dat'):
            os.remove('experiment_data.dat')
