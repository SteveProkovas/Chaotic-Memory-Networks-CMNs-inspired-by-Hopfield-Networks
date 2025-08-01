#!/usr/bin/env python3
"""
Safe Execution Wrapper for AMD Ryzen 5 Systems
- Prevents OOM crashes
- Manages thermal throttling
- Recovers from failures
- Maintains execution integrity
"""

import os
import sys
import time
import gc
import signal
import psutil
import logging
import traceback
import numpy as np
from functools import wraps
from memory_manager import HardwareResourceMonitor, MemoryAwareExecutor

# Configuration (should match hardware_config.ini)
DEFAULT_CONFIG = {
    'max_workers': 6,
    'memory_limit': 7.37 * 1024**3,  # 7.37GB usable
    'thermal_throttle_temp': 85,
    'mmap_threshold_mb': 200,
    'garbage_collection_interval': 30,
    'max_retries': 3,
    'retry_delay': 2.0,
    'cpu_throttle_threshold': 0.9,
    'memory_throttle_threshold': 0.85
}

class SafeExecutionWrapper:
    """System-safe execution environment with hardware monitoring"""
    def __init__(self, config=None, logger=None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logger or self._create_logger()
        self.monitor = HardwareResourceMonitor()
        self.executor = MemoryAwareExecutor(self.monitor)
        self.emergency_state = False
        self._setup_signal_handlers()
        
    def _create_logger(self):
        """Create a dedicated logger for safe execution"""
        logger = logging.getLogger('SafeExecutionWrapper')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler('safe_execution.log')
        fh.setLevel(logging.DEBUG)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def _setup_signal_handlers(self):
        """Handle system signals for graceful termination"""
        signal.signal(signal.SIGINT, self._handle_termination_signal)
        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        
        # Windows-specific signals
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._handle_termination_signal)
    
    def _handle_termination_signal(self, signum, frame):
        """Handle termination signals gracefully"""
        self.logger.warning(f"Received termination signal {signum}")
        self.emergency_state = True
        self.cleanup()
        sys.exit(1)
    
    def __enter__(self):
        """Context manager entry point"""
        self.monitor.start()
        self.logger.info("Safe execution environment activated")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point"""
        self.cleanup()
        if exc_type is not None:
            self.logger.error(f"Execution exited with error: {exc_value}")
            return False  # Propagate exception
        return True
    
    def cleanup(self):
        """Release all resources and clean up"""
        self.logger.info("Cleaning up resources")
        self.monitor.stop()
        
        # Clean up temporary files
        self._clean_temp_files()
        
        # Explicit garbage collection
        gc.collect()
        
        # Clear numpy cache
        try:
            np.savez_compressed('temp_cache_clear.npz', data=[])
            os.remove('temp_cache_clear.npz')
        except Exception:
            pass
        
        self.logger.info("Cleanup completed")
    
    def _clean_temp_files(self):
        """Remove temporary memory-mapped files"""
        temp_files = [f for f in os.listdir() if f.endswith('.dat')]
        for file in temp_files:
            try:
                os.remove(file)
                self.logger.debug(f"Removed temp file: {file}")
            except Exception as e:
                self.logger.warning(f"Couldn't remove {file}: {str(e)}")
    
    def system_check(self):
        """Verify system health before execution"""
        status = self.monitor.current_status()
        
        # Check critical conditions
        if status['temperature'] > self.config['thermal_throttle_temp']:
            self.logger.error(f"Critical temperature: {status['temperature']}°C")
            return False
            
        if status['memory_used'] > self.config['memory_throttle_threshold']:
            self.logger.error("Memory usage exceeds safe threshold")
            return False
            
        if status['cpu_used'] > self.config['cpu_throttle_threshold']:
            self.logger.warning("CPU usage exceeds threshold - throttling")
            
        return True
    
    def execute_safely(self, func, *args, **kwargs):
        """
        Execute a function with comprehensive safety mechanisms
        - Automatic retries
        - Memory management
        - Thermal protection
        - Resource monitoring
        """
        operation_type = kwargs.pop('operation_type', 'generic')
        pattern_size = kwargs.pop('pattern_size', 0)
        n_patterns = kwargs.pop('n_patterns', 1)
        max_retries = kwargs.pop('max_retries', self.config['max_retries'])
        
        for attempt in range(max_retries):
            try:
                # Pre-execution checks
                if not self.system_check():
                    self.logger.warning("System check failed - delaying execution")
                    time.sleep(self.config['retry_delay'] * (attempt + 1))
                    continue
                
                # Predict resource impact
                impact = self.executor.predict_operation_impact(
                    operation_type, 
                    pattern_size, 
                    n_patterns
                )
                self.logger.info(
                    f"Operation impact prediction: "
                    f"Current: {impact['current_usage']/1e6:.2f}MB, "
                    f"Predicted: {impact['predicted_usage']/1e6:.2f}MB, "
                    f"Peak: {impact['predicted_peak']/1e6:.2f}MB"
                )
                
                # Check if we have sufficient safety margin
                if impact['safety_margin'] < 0:
                    self.logger.warning(
                        "Insufficient safety margin: "
                        f"{impact['safety_margin']/1e6:.2f}MB deficit"
                    )
                    raise MemoryError("Predicted OOM condition")
                
                # Execute with monitoring
                result = self.executor.execute_and_record(
                    func, 
                    operation_type, 
                    pattern_size, 
                    n_patterns, 
                    *args, 
                    **kwargs
                )
                
                return result
                
            except MemoryError as e:
                self.logger.error(f"Memory error on attempt {attempt+1}: {str(e)}")
                self._handle_memory_error(attempt)
                
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt+1}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                self._handle_general_error(attempt, e)
        
        # If all retries failed
        raise RuntimeError(f"Operation failed after {max_retries} attempts")
    
    def _handle_memory_error(self, attempt):
        """Special handling for memory-related errors"""
        # Aggressive cleanup
        self.executor.monitor.force_memory_cleanup()
        gc.collect()
        
        # Increase delay with each attempt
        delay = self.config['retry_delay'] * (2 ** attempt)
        self.logger.info(f"Retrying after {delay:.1f} seconds")
        time.sleep(delay)
        
        # Reduce problem size if possible
        if hasattr(self, 'reduce_workload'):
            self.logger.info("Reducing workload size")
            self.reduce_workload(0.8)  # Reduce by 20%
    
    def _handle_general_error(self, attempt, exception):
        """Handle non-memory related exceptions"""
        # Special handling for numerical errors
        if 'float' in str(exception).lower():
            self.logger.warning("Numerical instability detected - reducing precision")
            self._reduce_precision()
        
        # Delay before retry
        delay = self.config['retry_delay'] * (attempt + 1)
        self.logger.info(f"Retrying after {delay:.1f} seconds")
        time.sleep(delay)
    
    def _reduce_precision(self):
        """Downgrade precision to conserve memory"""
        self.logger.info("Reducing numerical precision")
        # Implementation would depend on your specific application
        # Example: switch from float64 to float32, or float32 to float16
    
    def batch_executor(self, func, data, batch_size=None, operation_type="batch"):
        """
        Execute operations in safe batches
        - Automatic batch sizing
        - Progress monitoring
        - Memory-aware execution
        """
        if not batch_size:
            # Calculate safe batch size
            element_size = data[0].nbytes if hasattr(data[0], 'nbytes') else sys.getsizeof(data[0])
            batch_size = self.executor.enhanced_auto_batch_size(
                element_size,
                len(data),
                operation_type
            )
            self.logger.info(f"Auto-calculated batch size: {batch_size}")
        
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Check system health before each batch
            if not self.system_check():
                self.logger.warning("System overload - pausing batch processing")
                time.sleep(self.config['retry_delay'])
                
                # Reduce batch size dynamically
                batch_size = max(1, int(batch_size * 0.7))
                self.logger.info(f"Reduced batch size to {batch_size}")
            
            try:
                # Execute batch
                result = self.execute_safely(
                    func,
                    batch,
                    operation_type=operation_type,
                    pattern_size=len(batch[0]) if batch else 0,
                    n_patterns=len(batch)
                )
                results.append(result)
                
                # Progress logging
                self.logger.info(
                    f"Processed batch {i//batch_size + 1}/"
                    f"{int(np.ceil(len(data)/batch_size))} "
                    f"({min(i+batch_size, len(data))}/{len(data)} items)"
                )
                
            except Exception as e:
                self.logger.error(f"Batch processing failed: {str(e)}")
                # Save partial results
                self.logger.warning("Returning partial results")
                break
        
        return results
    
    def resource_throttle(self, func):
        """Decorator to throttle function based on resource usage"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check CPU load
            while psutil.cpu_percent() > self.config['cpu_throttle_threshold'] * 100:
                self.logger.warning("CPU throttling active - waiting")
                time.sleep(0.5)
            
            # Check memory pressure
            mem = psutil.virtual_memory()
            while mem.percent > self.config['memory_throttle_threshold'] * 100:
                self.logger.warning("Memory throttling active - waiting")
                time.sleep(1.0)
                mem = psutil.virtual_memory()
            
            # Execute function
            return func(*args, **kwargs)
        return wrapper


# Example Usage
if __name__ == "__main__":
    # Example memory-intensive operation
    def matrix_operation(data):
        """Simulate a memory-intensive operation"""
        result = np.dot(data, data.T)
        # Simulate computation time
        time.sleep(0.1 * len(data))
        return result
    
    # Create test data
    data = [np.random.rand(500, 500).astype(np.float32) for _ in range(20)]
    
    with SafeExecutionWrapper() as wrapper:
        try:
            # Execute single operation safely
            result = wrapper.execute_safely(
                matrix_operation,
                data[0],
                operation_type="matrix_operation",
                pattern_size=500*500,
                n_patterns=1
            )
            print(f"Operation result shape: {result.shape}")
            
            # Execute batch processing
            results = wrapper.batch_executor(
                matrix_operation,
                data,
                operation_type="batch_matrix_operation"
            )
            print(f"Processed {len(results)} batches")
            
            # Throttled function example
            @wrapper.resource_throttle
            def critical_function(x):
                return np.linalg.inv(x)
            
            for matrix in data:
                inv = critical_function(matrix)
                print(f"Inverse shape: {inv.shape}")
                
        except Exception as e:
            print(f"Critical failure: {str(e)}")
