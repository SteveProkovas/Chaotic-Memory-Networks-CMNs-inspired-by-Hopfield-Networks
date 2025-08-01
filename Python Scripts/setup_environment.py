#!/usr/bin/env python3
"""
Hardware-Optimized Environment Setup for AMD Ryzen 5 4600H
- Configures system for optimal chaotic memory network experiments
- Implements AMD-specific performance tweaks
- Ensures resource-efficient operation within 8GB RAM constraints
"""

import os
import sys
import psutil
import platform
import subprocess
import configparser
import numpy as np
from datetime import datetime

# Configuration constants
CONFIG_FILE = "hardware_config.ini"
LOG_FILE = "environment_setup.log"
REQUIRED_PACKAGES = [
    'numpy', 'scipy', 'psutil', 'scikit-learn', 'matplotlib',
    'h5py', 'numba', 'tqdm', 'py-cpuinfo'
]
AMD_OPTIMIZED_LIBS = {
    'numpy': '1.25.0',
    'scipy': '1.11.0'
}

class RyzenOptimizer:
    """AMD Ryzen-specific performance optimizer"""
    def __init__(self):
        self.system_info = self.get_system_info()
        self.config = self.load_or_create_config()
        self.log("Initializing Ryzen Optimizer")
        
    def get_system_info(self):
        """Collect detailed hardware specifications"""
        cpu_info = self.get_cpu_info()
        mem_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.architecture()[0]
        }
        
        return {
            "cpu": cpu_info,
            "memory": {
                "total": mem_info.total,
                "available": mem_info.available,
                "used": mem_info.used,
                "free": mem_info.free,
                "percent": mem_info.percent
            },
            "swap": {
                "total": swap_info.total,
                "used": swap_info.used,
                "free": swap_info.free,
                "percent": swap_info.percent
            },
            "os": os_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cpu_info(self):
        """Get detailed CPU information with AMD-specific features"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return {
                "brand_raw": info['brand_raw'],
                "hz_actual": info['hz_actual'],
                "hz_advertised": info['hz_advertised'],
                "cores_physical": info['count'],
                "cores_logical": psutil.cpu_count(logical=True),
                "l1_cache_size": info.get('l1_data_cache_size', 32768),
                "l2_cache_size": info.get('l2_cache_size', 262144),
                "l3_cache_size": info.get('l3_cache_size', 8388608),
                "amd_features": [f for f in info['flags'] if 'amd' in f.lower()]
            }
        except ImportError:
            return {
                "brand_raw": "AMD Ryzen 5 4600H",
                "hz_actual": 3.0e9,
                "cores_physical": 6,
                "cores_logical": 12
            }
    
    def load_or_create_config(self):
        """Load or create hardware-specific configuration"""
        config = configparser.ConfigParser()
        
        if os.path.exists(CONFIG_FILE):
            config.read(CONFIG_FILE)
            self.log(f"Loaded existing config: {CONFIG_FILE}")
        else:
            self.log("Creating new hardware configuration")
            self.create_default_config(config)
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
        
        return config
    
    def create_default_config(self, config):
        """Create AMD-optimized default configuration"""
        # System Settings
        config['System'] = {
            'max_workers': str(self.system_info['cpu']['cores_logical']),
            'memory_limit': str(self.system_info['memory']['total']),
            'swap_usage': 'avoid',  # Prefer RAM over swap
            'thermal_throttle_temp': '85'  # Ryzen throttling point
        }
        
        # AMD-Specific Optimizations
        config['AMD_Optimizations'] = {
            'use_avx2': 'true',
            'use_fma3': 'true',
            'numa_optimized': 'true',
            'ccx_aware_scheduling': 'true',
            'prefetch_distance': '64',
            'l2_cache_partitioning': 'auto'
        }
        
        # Memory Management
        config['Memory'] = {
            'default_precision': 'float32',
            'auto_downgrade_precision': 'true',
            'mmap_threshold_mb': '200',  # Use memmap for arrays >200MB
            'garbage_collection_interval': '30'
        }
        
        # Experiment Settings
        config['Experiments'] = {
            'max_patterns': '150',
            'max_neurons': '1000',
            'mnist_downscale': '8x8',
            'batch_size_heuristic': 'conservative'
        }
        
        # Visualization Settings
        config['Visualization'] = {
            'interactive_backend': 'TkAgg' if self.system_info['os']['system'] == 'Windows' else 'Agg',
            'dpi_scaling': '0.75'
        }
        
        # Logging
        config['Logging'] = {
            'level': 'INFO',
            'max_size_mb': '10',
            'backup_count': '3'
        }
    
    def apply_environment_variables(self):
        """Set environment variables for optimized performance"""
        # Set NumPy configuration
        os.environ['NPY_NUM_BUFSIZE'] = '8192'  # Buffer size for AMD
        os.environ['NPY_NUM_THREADS'] = self.config['System']['max_workers']
        os.environ['OMP_NUM_THREADS'] = self.config['System']['max_workers']
        
        # AMD-specific math library optimizations
        os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
        os.environ['NUMBA_ENABLE_AVX'] = '1'
        
        # Memory management settings
        os.environ['PYTHONMALLOC'] = 'malloc'
        os.environ['PYTHONFAULTHANDLER'] = '1'
        
        # Configure garbage collection
        gc_settings = self.config['Memory']
        os.environ['PYTHONGCINTERVAL'] = gc_settings['garbage_collection_interval']
        
        self.log("Environment variables configured")
    
    def install_packages(self):
        """Install required packages with AMD-optimized versions"""
        self.log("Checking package dependencies")
        
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        pip_command = [sys.executable, '-m', 'pip']
        
        # Install AMD-optimized libraries
        for package, version in AMD_OPTIMIZED_LIBS.items():
            self.log(f"Installing AMD-optimized {package}=={version}")
            subprocess.run(
                pip_command + ['install', f'{package}=={version}'],
                check=True
            )
        
        # Install other requirements
        for package in REQUIRED_PACKAGES:
            if package not in AMD_OPTIMIZED_LIBS:
                self.log(f"Installing {package}")
                subprocess.run(
                    pip_command + ['install', package],
                    check=True
                )
        
        # Install AMD-specific math libraries if available
        if self.system_info['os']['system'] == 'Linux':
            try:
                self.log("Installing AMD Optimized BLAS (blis)")
                subprocess.run(
                    pip_command + ['install', 'blis'],
                    check=True
                )
            except subprocess.CalledProcessError:
                self.log("Failed to install blis, using OpenBLAS fallback")
        
        self.log("Package installation complete")
    
    def configure_num_threads(self):
        """Set optimal thread count for AMD architecture"""
        logical_cores = self.system_info['cpu']['cores_logical']
        physical_cores = self.system_info['cpu']['cores_physical']
        
        # Set thread counts in configuration
        self.config['System']['max_workers'] = str(physical_cores)  # Use physical cores for CPU-bound
        self.config['System']['io_workers'] = str(logical_cores - physical_cores)  # Use SMT for I/O
        
        # Update environment variables
        os.environ['OMP_NUM_THREADS'] = str(physical_cores)
        os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(physical_cores)
        
        self.log(f"Configured {physical_cores} compute threads and {logical_cores - physical_cores} I/O threads")
    
    def optimize_numpy(self):
        """Apply AMD-specific NumPy optimizations"""
        import numpy as np
        from numpy.core import _exceptions
        
        # Set array printing options for reduced memory
        np.set_printoptions(precision=4, threshold=100, edgeitems=3, linewidth=140)
        
        # Configure array allocation strategy
        np.core.multiarray._set_memory_policy('page')
        
        # AMD-specific optimizations
        if 'avx2' in self.system_info['cpu']['amd_features']:
            _exceptions._set_umath_flags(flags=['AVX2'], env=os.environ)
        
        self.log("NumPy optimized for AMD architecture")
    
    def create_directory_structure(self):
        """Create project directory structure"""
        dirs = [
            'data/raw',
            'data/processed',
            'experiments/results',
            'models/checkpoints',
            'logs',
            'notebooks',
            'src/utils',
            'src/core',
            'docs'
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            self.log(f"Created directory: {directory}")
        
        # Create empty placeholder files
        open('src/__init__.py', 'a').close()
        open('src/utils/__init__.py', 'a').close()
        open('src/core/__init__.py', 'a').close()
    
    def configure_swappiness(self):
        """Reduce swappiness to prioritize RAM usage"""
        if self.system_info['os']['system'] == 'Linux':
            try:
                # Read current swappiness
                with open('/proc/sys/vm/swappiness', 'r') as f:
                    current = int(f.read().strip())
                
                if current > 10:
                    self.log(f"Reducing swappiness from {current} to 10")
                    subprocess.run(['sudo', 'sysctl', 'vm.swappiness=10'], check=True)
                    
                    # Make permanent
                    with open('/etc/sysctl.conf', 'a') as f:
                        f.write('\n# AMD Optimization - Reduce swap usage\nvm.swappiness=10\n')
                else:
                    self.log(f"Swappiness already optimized at {current}")
            except Exception as e:
                self.log(f"Couldn't adjust swappiness: {str(e)}")
        else:
            self.log("Swappiness adjustment only available on Linux")
    
    def set_cpu_governor(self):
        """Set CPU governor to performance mode"""
        if self.system_info['os']['system'] == 'Linux':
            try:
                self.log("Setting CPU governor to performance mode")
                for cpu in range(self.system_info['cpu']['cores_logical']):
                    path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                    if os.path.exists(path):
                        with open(path, 'w') as f:
                            f.write('performance')
            except Exception as e:
                self.log(f"Couldn't set CPU governor: {str(e)}")
        else:
            self.log("CPU governor adjustment only available on Linux")
    
    def run_system_checks(self):
        """Perform hardware validation checks"""
        # Memory check
        mem_gb = self.system_info['memory']['total'] / (1024 ** 3)
        if mem_gb < 7:
            self.log(f"WARNING: Only {mem_gb:.1f}GB RAM available - experiments will be constrained")
        
        # CPU feature check
        cpu = self.system_info['cpu']
        if 'avx2' not in cpu.get('amd_features', []):
            self.log("WARNING: AVX2 instructions not available - performance may suffer")
        
        # Thermal check
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_temp = gpus[0].temperature
                if gpu_temp > 70:
                    self.log(f"WARNING: High GPU temperature detected: {gpu_temp}°C")
        except ImportError:
            pass
    
    def log(self, message):
        """Log setup messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Print to console
        print(log_entry)
        
        # Append to log file
        with open(LOG_FILE, 'a') as logfile:
            logfile.write(log_entry + '\n')
    
    def finalize_setup(self):
        """Complete setup process and save configuration"""
        # Save updated configuration
        with open(CONFIG_FILE, 'w') as configfile:
            self.config.write(configfile)
        
        self.log("Environment setup completed successfully")
        self.log("System configuration summary:")
        self.log(f"  CPU: {self.system_info['cpu']['brand_raw']}")
        self.log(f"  Cores: {self.system_info['cpu']['cores_physical']}P/{self.system_info['cpu']['cores_logical']}L")
        self.log(f"  RAM: {self.system_info['memory']['total'] / (1024**3):.2f} GB")
        self.log(f"  OS: {self.system_info['os']['system']} {self.system_info['os']['release']}")
        
        # Print important settings
        self.log("\nKey configuration settings:")
        self.log(f"  Max compute threads: {self.config['System']['max_workers']}")
        self.log(f"  Default precision: {self.config['Memory']['default_precision']}")
        self.log(f"  Max patterns: {self.config['Experiments']['max_patterns']}")
        self.log(f"  Memory-mapped threshold: {self.config['Memory']['mmap_threshold_mb']} MB")

    def execute(self):
        """Run full setup sequence"""
        self.log("===== Starting AMD-Optimized Environment Setup =====")
        self.run_system_checks()
        self.install_packages()
        self.configure_num_threads()
        self.apply_environment_variables()
        self.optimize_numpy()
        self.create_directory_structure()
        
        # Linux-specific optimizations
        if self.system_info['os']['system'] == 'Linux':
            self.configure_swappiness()
            self.set_cpu_governor()
        
        self.finalize_setup()


if __name__ == "__main__":
    optimizer = RyzenOptimizer()
    optimizer.execute()
