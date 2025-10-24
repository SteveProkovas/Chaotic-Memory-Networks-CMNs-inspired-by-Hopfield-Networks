import numpy as np
import mmap
import os
import tempfile
import threading
import queue
import time
from typing import Iterator, Generator, Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import pickle
import json

from core_cmn import ChaoticMemoryNetwork, HardwareConfig, ResourceManager

logger = logging.getLogger(__name__)

class MemoryMappedDataset:
    """
    Memory-mapped dataset for large-scale CMN experiments
    Optimized for AMD Ryzen 5 4600H with 8GB RAM
    """
    
    def __init__(self, filepath: str, mode: str = 'r+', 
                 chunk_size: int = 1024, max_memory_mb: int = 1000):
        """
        Initialize memory-mapped dataset
        
        Args:
            filepath: Path to data file
            mode: File access mode ('r', 'r+', 'w+')
            chunk_size: Size of data chunks for processing
            max_memory_mb: Maximum memory usage in MB
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.mmap_file = None
        self.data_array = None
        self.current_position = 0
        
        # Performance monitoring
        self.access_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Memory-mapped dataset initialized: {filepath}")
        logger.info(f"Chunk size: {chunk_size}, Max memory: {max_memory_mb}MB")
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def open(self):
        """Open memory-mapped file"""
        try:
            if not self.filepath.exists():
                # Create file if it doesn't exist
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                # Create a temporary file for testing
                temp_data = np.random.randn(1000, 100).astype(np.float32)
                np.save(self.filepath, temp_data)
            
            # Load data
            self.data_array = np.load(self.filepath, mmap_mode=self.mode)
            logger.info(f"Memory-mapped file opened: {self.data_array.shape}")
            
        except Exception as e:
            logger.error(f"Failed to open memory-mapped file: {e}")
            raise
    
    def close(self):
        """Close memory-mapped file"""
        if self.data_array is not None:
            del self.data_array
            self.data_array = None
        logger.info("Memory-mapped file closed")
    
    def get_chunk(self, start_idx: int = None, end_idx: int = None) -> np.ndarray:
        """
        Get a chunk of data from the memory-mapped file
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            
        Returns:
            Data chunk as numpy array
        """
        if self.data_array is None:
            raise RuntimeError("Dataset not opened")
        
        if start_idx is None:
            start_idx = self.current_position
        
        if end_idx is None:
            end_idx = min(start_idx + self.chunk_size, len(self.data_array))
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(self.data_array))
        
        if start_idx >= end_idx:
            return np.array([])
        
        # Get chunk
        chunk = self.data_array[start_idx:end_idx]
        self.current_position = end_idx
        self.access_count += 1
        
        logger.debug(f"Retrieved chunk [{start_idx}:{end_idx}], shape: {chunk.shape}")
        return chunk
    
    def get_streaming_iterator(self, batch_size: int = None) -> Iterator[np.ndarray]:
        """
        Get streaming iterator for batch processing
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Data batches
        """
        if batch_size is None:
            batch_size = self.chunk_size
        
        current_idx = 0
        total_samples = len(self.data_array)
        
        while current_idx < total_samples:
            end_idx = min(current_idx + batch_size, total_samples)
            batch = self.get_chunk(current_idx, end_idx)
            
            if len(batch) > 0:
                yield batch
            
            current_idx = end_idx
            
            # Memory management
            if self.access_count % 10 == 0:
                self._check_memory_usage()
    
    def _check_memory_usage(self):
        """Check and manage memory usage"""
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 80:  # 80% threshold
            logger.warning(f"High memory usage: {memory_usage:.1f}%")
            # Force garbage collection
            import gc
            gc.collect()

class StreamingCMN:
    """
    Streaming Chaotic Memory Network with memory-mapped data support
    """
    
    def __init__(self, n_neurons: int, dataset_path: str,
                 hardware_config: Optional[HardwareConfig] = None):
        """
        Initialize streaming CMN
        
        Args:
            n_neurons: Number of neurons
            dataset_path: Path to memory-mapped dataset
            hardware_config: Hardware configuration
        """
        self.n_neurons = n_neurons
        self.dataset_path = dataset_path
        self.config = hardware_config or HardwareConfig()
        
        # Initialize base CMN
        self.cmn = ChaoticMemoryNetwork(n_neurons, hardware_config=self.config)
        
        # Streaming components
        self.data_queue = queue.Queue(maxsize=100)
        self.processing_threads = []
        self.is_streaming = False
        
        # Performance metrics
        self.streaming_metrics = {
            'patterns_processed': 0,
            'processing_time': [],
            'memory_usage': [],
            'throughput': []
        }
        
        logger.info(f"Streaming CMN initialized with {n_neurons} neurons")
        logger.info(f"Dataset path: {dataset_path}")
    
    def start_streaming_processing(self, batch_size: int = 32, 
                                 max_workers: int = None) -> None:
        """
        Start streaming data processing
        
        Args:
            batch_size: Size of data batches
            max_workers: Maximum number of worker threads
        """
        if max_workers is None:
            max_workers = min(self.config.cpu_cores, 4)  # Limit to 4 for stability
        
        self.is_streaming = True
        
        # Start data loading thread
        data_thread = threading.Thread(
            target=self._data_loading_worker,
            args=(batch_size,)
        )
        data_thread.daemon = True
        data_thread.start()
        
        # Start processing threads
        for i in range(max_workers):
            process_thread = threading.Thread(
                target=self._processing_worker,
                args=(f"worker-{i}",)
            )
            process_thread.daemon = True
            process_thread.start()
            self.processing_threads.append(process_thread)
        
        logger.info(f"Streaming processing started with {max_workers} workers")
    
    def stop_streaming_processing(self) -> None:
        """Stop streaming data processing"""
        self.is_streaming = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        logger.info("Streaming processing stopped")
    
    def _data_loading_worker(self, batch_size: int) -> None:
        """Data loading worker thread"""
        try:
            with MemoryMappedDataset(self.dataset_path) as dataset:
                for batch in dataset.get_streaming_iterator(batch_size):
                    if not self.is_streaming:
                        break
                    
                    # Add batch to queue
                    self.data_queue.put(batch, timeout=1.0)
                    
        except Exception as e:
            logger.error(f"Data loading worker error: {e}")
    
    def _processing_worker(self, worker_name: str) -> None:
        """Data processing worker thread"""
        logger.info(f"Processing worker {worker_name} started")
        
        while self.is_streaming:
            try:
                # Get batch from queue
                batch = self.data_queue.get(timeout=1.0)
                
                # Process batch
                start_time = time.time()
                success_count = self._process_batch(batch)
                processing_time = time.time() - start_time
                
                # Update metrics
                self.streaming_metrics['patterns_processed'] += len(batch)
                self.streaming_metrics['processing_time'].append(processing_time)
                
                # Memory monitoring
                import psutil
                memory_usage = psutil.virtual_memory().percent
                self.streaming_metrics['memory_usage'].append(memory_usage)
                
                logger.debug(f"{worker_name}: Processed {len(batch)} patterns, "
                           f"success: {success_count}, time: {processing_time:.4f}s")
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker {worker_name} error: {e}")
    
    def _process_batch(self, batch: np.ndarray) -> int:
        """
        Process a batch of patterns
        
        Args:
            batch: Batch of patterns to process
            
        Returns:
            Number of successfully processed patterns
        """
        success_count = 0
        
        for pattern in batch:
            try:
                # Store pattern in CMN
                success = self.cmn.store_pattern(pattern, strength=1.0)
                if success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Pattern processing error: {e}")
        
        return success_count
    
    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics"""
        if not self.streaming_metrics['processing_time']:
            return self.streaming_metrics
        
        avg_processing_time = np.mean(self.streaming_metrics['processing_time'])
        avg_memory_usage = np.mean(self.streaming_metrics['memory_usage'])
        
        # Calculate throughput
        total_time = sum(self.streaming_metrics['processing_time'])
        throughput = self.streaming_metrics['patterns_processed'] / total_time if total_time > 0 else 0
        
        return {
            'patterns_processed': self.streaming_metrics['patterns_processed'],
            'average_processing_time': avg_processing_time,
            'average_memory_usage': avg_memory_usage,
            'throughput_patterns_per_second': throughput,
            'worker_threads': len(self.processing_threads),
            'queue_size': self.data_queue.qsize()
        }
    
    def save_streaming_state(self, filepath: str) -> bool:
        """Save streaming CMN state"""
        try:
            state = {
                'cmn_state': self.cmn.get_performance_metrics(),
                'streaming_metrics': self.streaming_metrics,
                'config': {
                    'n_neurons': self.n_neurons,
                    'dataset_path': self.dataset_path
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Streaming state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save streaming state: {e}")
            return False

class FaultTolerantCMN:
    """
    Fault-tolerant CMN with recovery protocols for numerical stability
    """
    
    def __init__(self, n_neurons: int, 
                 hardware_config: Optional[HardwareConfig] = None):
        """
        Initialize fault-tolerant CMN
        
        Args:
            n_neurons: Number of neurons
            hardware_config: Hardware configuration
        """
        self.n_neurons = n_neurons
        self.config = hardware_config or HardwareConfig()
        
        # Initialize base CMN
        self.cmn = ChaoticMemoryNetwork(n_neurons, hardware_config=self.config)
        
        # Fault tolerance components
        self.checkpoint_interval = 100  # Checkpoint every 100 operations
        self.operation_count = 0
        self.last_checkpoint = None
        
        # Recovery protocols
        self.recovery_protocols = {
            'numerical_overflow': self._handle_numerical_overflow,
            'memory_exhaustion': self._handle_memory_exhaustion,
            'convergence_failure': self._handle_convergence_failure
        }
        
        logger.info(f"Fault-tolerant CMN initialized with {n_neurons} neurons")
    
    def store_pattern_with_recovery(self, pattern: np.ndarray, 
                                  strength: float = 1.0) -> bool:
        """
        Store pattern with fault tolerance and recovery
        
        Args:
            pattern: Pattern to store
            strength: Storage strength
            
        Returns:
            Success status
        """
        try:
            # Check for numerical issues
            if np.any(np.isnan(pattern)) or np.any(np.isinf(pattern)):
                logger.warning("Pattern contains NaN or Inf values, applying recovery")
                pattern = self._recover_pattern(pattern)
            
            # Store pattern
            success = self.cmn.store_pattern(pattern, strength)
            
            if success:
                self.operation_count += 1
                
                # Create checkpoint if needed
                if self.operation_count % self.checkpoint_interval == 0:
                    self._create_checkpoint()
            
            return success
            
        except Exception as e:
            logger.error(f"Pattern storage failed: {e}")
            return self._attempt_recovery('storage_failure', pattern, strength)
    
    def recall_pattern_with_recovery(self, partial_pattern: np.ndarray,
                                   max_iterations: int = 100) -> Tuple[np.ndarray, bool]:
        """
        Recall pattern with fault tolerance
        
        Args:
            partial_pattern: Partial pattern for recall
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (recalled_pattern, success)
        """
        try:
            # Check for numerical issues
            if np.any(np.isnan(partial_pattern)) or np.any(np.isinf(partial_pattern)):
                logger.warning("Partial pattern contains NaN or Inf values")
                partial_pattern = self._recover_pattern(partial_pattern)
            
            # Attempt recall
            recalled, success = self.cmn.recall_pattern(partial_pattern, max_iterations)
            
            if not success:
                logger.warning("Recall failed, attempting recovery")
                return self._attempt_recall_recovery(partial_pattern, max_iterations)
            
            return recalled, success
            
        except Exception as e:
            logger.error(f"Pattern recall failed: {e}")
            return np.zeros_like(partial_pattern), False
    
    def _recover_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Recover pattern from numerical issues"""
        # Replace NaN and Inf with zeros
        recovered = pattern.copy()
        recovered[np.isnan(recovered)] = 0.0
        recovered[np.isinf(recovered)] = 0.0
        
        # Normalize to prevent overflow
        if np.max(np.abs(recovered)) > 10.0:
            recovered = np.tanh(recovered)
        
        return recovered
    
    def _create_checkpoint(self) -> None:
        """Create system checkpoint"""
        try:
            checkpoint_data = {
                'weights': self.cmn.weights.copy(),
                'neurons': self.cmn.neurons.copy(),
                'chaos_state': self.cmn.chaos_state.copy(),
                'operation_count': self.operation_count,
                'timestamp': time.time()
            }
            
            self.last_checkpoint = checkpoint_data
            logger.info(f"Checkpoint created at operation {self.operation_count}")
            
        except Exception as e:
            logger.error(f"Checkpoint creation failed: {e}")
    
    def _attempt_recovery(self, error_type: str, *args) -> bool:
        """Attempt system recovery"""
        if error_type in self.recovery_protocols:
            return self.recovery_protocols[error_type](*args)
        return False
    
    def _handle_numerical_overflow(self, pattern: np.ndarray, strength: float) -> bool:
        """Handle numerical overflow recovery"""
        logger.info("Applying numerical overflow recovery")
        
        # Reduce chaos parameter
        self.cmn.chaos_param *= 0.8
        
        # Normalize weights
        self.cmn.weights = np.tanh(self.cmn.weights)
        
        # Retry with reduced strength
        return self.cmn.store_pattern(pattern, strength * 0.5)
    
    def _handle_memory_exhaustion(self, pattern: np.ndarray, strength: float) -> bool:
        """Handle memory exhaustion recovery"""
        logger.info("Applying memory exhaustion recovery")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Reduce precision
        pattern = pattern.astype(np.float32)
        
        # Retry
        return self.cmn.store_pattern(pattern, strength)
    
    def _handle_convergence_failure(self, partial_pattern: np.ndarray, 
                                  max_iterations: int) -> Tuple[np.ndarray, bool]:
        """Handle convergence failure recovery"""
        logger.info("Applying convergence failure recovery")
        
        # Increase chaos parameter
        self.cmn.chaos_param *= 1.2
        
        # Reduce tolerance
        tolerance = 1e-4  # More lenient tolerance
        
        # Retry with adjusted parameters
        return self.cmn.recall_pattern(partial_pattern, max_iterations, tolerance)
    
    def _attempt_recall_recovery(self, partial_pattern: np.ndarray,
                                max_iterations: int) -> Tuple[np.ndarray, bool]:
        """Attempt recall recovery"""
        # Try with reduced chaos parameter
        original_chaos = self.cmn.chaos_param
        self.cmn.chaos_param *= 0.8
        
        recalled, success = self.cmn.recall_pattern(partial_pattern, max_iterations)
        
        if not success:
            # Restore original chaos parameter
            self.cmn.chaos_param = original_chaos
            return np.zeros_like(partial_pattern), False
        
        return recalled, success

# Example usage
if __name__ == "__main__":
    # Create fault-tolerant CMN
    ft_cmn = FaultTolerantCMN(n_neurons=100)
    
    # Test with problematic patterns
    test_patterns = [
        np.random.randn(100),  # Normal pattern
        np.random.randn(100) * 100,  # Large values
        np.random.randn(100) + np.nan,  # NaN values
    ]
    
    print("Testing fault-tolerant CMN...")
    
    for i, pattern in enumerate(test_patterns):
        print(f"\\nTesting pattern {i+1}:")
        success = ft_cmn.store_pattern_with_recovery(pattern)
        print(f"Storage: {'Success' if success else 'Failed'}")
        
        if success:
            # Test recall
            noisy_pattern = pattern + 0.3 * np.random.randn(100)
            recalled, recall_success = ft_cmn.recall_pattern_with_recovery(noisy_pattern)
            print(f"Recall: {'Success' if recall_success else 'Failed'}")
            
            if recall_success:
                correlation = np.corrcoef(pattern, recalled)[0, 1]
                print(f"Correlation: {correlation:.4f}")
    
    print("\\nFault-tolerant CMN testing completed!")
