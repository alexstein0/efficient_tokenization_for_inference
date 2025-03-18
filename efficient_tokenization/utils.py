import os
import shutil
import logging
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
import torch
import json
import hashlib
import psutil

def setup_logging(log_level=logging.INFO):
    """Setup global logging configuration"""
    # Set up basic configuration first
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=log_level,
        force=True  # This ensures we override any existing configuration
    )
    try:
        ll = logging.getLevelName(log_level)
        logger = get_logger(__name__)
        logger.setLevel(ll)
        logger.info("Initialized logger")
    except:
        logger = logging.getLogger(__name__)  # if you want to see for all ranks
        logger.info("AcceleratorState not initialized")
    
    return logger

logger = logging.getLogger(__name__)  # if you want to see for all ranks

def get_directory_size(directory):
    """Calculate the total size of a directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


def format_size(bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024


def check_disk_space(directory, required_space_gb=10):
    """
    Check if there's enough disk space to save the model.
    
    Args:
        directory: Directory where model will be saved
        accelerator: Accelerator instance for distributed training logging
        required_space_gb: Required free space in GB (default 10GB)
    
    Returns:
        bool: True if enough space available, False otherwise
    """
    try:
        # Get free space in bytes
        free_space = shutil.disk_usage(directory).free
        required_space = required_space_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        
        # If directory exists, add its current size to required space
        if os.path.exists(directory):
            required_space += get_directory_size(directory)
        
        if free_space < required_space:
            logger.warning(f"Warning: Not enough disk space!")
            logger.warning(f"Available: {format_size(free_space)}")
            logger.warning(f"Required: {format_size(required_space)}")
            return False
        
        logger.debug(f"Sufficient disk space available:")
        logger.debug(f"Free space: {format_size(free_space)}")
        logger.debug(f"Required space: {format_size(required_space)}")
        return True
        
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return False

def log_memory_usage(step, phase, accelerator):
    """Log memory usage at various points in training"""
    if accelerator.is_local_main_process:
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"Step {step} - {phase} - "
                   f"GPU Memory Allocated: {gpu_memory_allocated:.2f}MB, "
                   f"Reserved: {gpu_memory_reserved:.2f}MB")


def generate_hashed_dir_name(params_dict, output_folder="output", dry_run=False):
    # Serialize the dictionary of parameters in a deterministic way
    params_json = json.dumps(params_dict, sort_keys=True).encode()
    params_hash = hashlib.md5(params_json).hexdigest()[:8]  # 8 chars short hash

    output_dir = f"{params_hash}-{params_dict['model_name']}-{params_dict['task_name']}-{params_dict['num_new_tokens']}"
    
    if dry_run:
        output_dir = f"dryrun-{output_dir}"
    
    return os.path.join(output_folder, output_dir)

def get_cpus() -> int:
    # Number of threads
    try:
        return min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))  # covering both affinity and phys.
    except:
        pass
    try:
        return os.cpu_count()  # when running on mac
    except:
        return 1

