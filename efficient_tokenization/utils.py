import os
import shutil
import logging
from accelerate.logging import get_logger
import torch
import json
import hashlib
import psutil
import argparse
import glob
import numpy as np
from typing import Dict, Tuple, List


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


def generate_hashed_dir_name(params_dict: Dict, dry_run=False):
    # Serialize the dictionary of parameters in a deterministic way
    params_json = json.dumps(params_dict, sort_keys=True).encode()
    params_hash = hashlib.md5(params_json).hexdigest()[:8]  # 8 chars short hash

    output_dir = f"{params_hash}-{params_dict['model_name']}-{params_dict['task_name']}"
    # if include_num_tokens:
    output_dir = f"{output_dir}-{params_dict['num_new_tokens']}"
    
    if dry_run:
        output_dir = f"dryrun-{output_dir}"
    
    return output_dir

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
    
def get_latest_checkpoint(output_dir: str, recursively_check = True) -> Tuple[str| None, List[str]]:
    """Find the latest checkpoint in the checkpoints directory."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoints_to_remove = []
    if not os.path.exists(checkpoint_dir):
        # logger.warning(f"No checkpoints directory found at {checkpoint_dir}")
        return None, checkpoints_to_remove
    
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))
    if not checkpoint_paths:
        # logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None, checkpoints_to_remove
    
    # Sort by modification time (most recent first)
    checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
    latest = None
    for path in checkpoint_paths:
        if os.path.exists(os.path.join(path, "checkpoint_meta.pt")):
            latest = path
            logger.info(f"Found latest checkpoint: {latest}")
            break
        else:
            logger.info(f"Checkpoint {path} does not have checkpoint_meta.pt {'skipping' if recursively_check else ''}")
            if not recursively_check:
                return path, checkpoints_to_remove
            checkpoints_to_remove.append(path)

    return latest, checkpoints_to_remove


def parse_args():
    threads = get_cpus()

    args = argparse.ArgumentParser()

    # implementation params
    args.add_argument("--experiment-name", type=str, default="default")
    args.add_argument("--extra-info", type=str, default=None)
    args.add_argument("--dry-run", action="store_true")
    args.add_argument("--logging-mode", type=str, choices=["INFO", "DEBUG"], default="INFO")
    args.add_argument("--cpu-batch-size", type=int, default=1000)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--overwrite-final", action="store_true", help="Before running we check if final model already exists and if it does we require this flag to not risk overwriting")
    args.add_argument("--resume-from-checkpoint", type=str, default="latest",
                     help="Path to checkpoint directory or 'latest' to let accelerate handle latest")
    args.add_argument("--wandb", type=str)
    args.add_argument("--wandb-tags", type=str)
    args.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility. This seed is also used to create the output directory so different runs with the same seed will overwrite each other")
    args.add_argument("--output-dir", type=str, required=False, default=None)
    args.add_argument("--num-proc", type=int, default=threads)
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--save-dataset-path", type=str)
    args.add_argument("--save-checkpoints", type=str,
                    choices=["final", "all", None, "model_only"],
                    default="model_only",
                    help="Whether to save the model after training")
    args.add_argument("--fsdp", action="store_true")

    # finetuning params
    args.add_argument("--dataset", type=str, default=None)
    args.add_argument("--dataset-dir", type=str, default="datasets")
    args.add_argument("--total-batch-size", type=int, default=None)
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--max-train-steps", type=int, default=-1)
    args.add_argument("--num-epochs", type=int, default=-1)
    args.add_argument("--grad-norm", type=float, default=1.0, help="Max norm for gradient clipping. Set to None to disable.")
    args.add_argument("--deepspeed", action="store_true")  # TODO
    args.add_argument("--task-name", type=str,
                      choices=["SFT", "translation", "mixed"],
                      default="SFT",
                      help="Whether to finetune the model parameters")
    args.add_argument("--unfreeze-params-steps", type=int,
                    default=0,
                    help="Steps to switch to finetuning params")
    args.add_argument("--finetune-params", type=str,
                      choices=["full", "embeddings", "new_tokens_only", "lora", "first_last"],
                      default="full",
                      help="Whether to finetune the model parameters after unfreezing")
    args.add_argument("--finetune-params-prefreeze", type=str,
                      choices=["full", "embeddings", "new_tokens_only", "lora", "first_last"],
                      default="full",
                      help="Whether to finetune the model parameters")
    args.add_argument("--reset-optimizer", action="store_true", default=False,
                      help="Whether to reset the optimizer after unfreezing")
    args.add_argument("--log-step", type=int, default=0,
                      help="step number for logging purposes.  Defaults to 0, but can be overridden by checkpoint loading")

    # lora params
    args.add_argument("--lora-target-modules", type=str, default="linear")
    args.add_argument("--lora-r", type=int, default=16)
    args.add_argument("--lora-alpha", type=int, default=64)
    args.add_argument("--lora-dropout", type=float, default=0.05)
    
    args.add_argument("--main-loss", type=str,
                      choices=["all", "translated", "new_tokens", None],
                      default=None,
                      help="Whether to backpropagate on all losses or just some")
    args.add_argument("--train-losses-to-track", type=str,
                      default=None,
                      help="List of losses to track during training")
    args.add_argument("--eval-losses-to-track", type=str,
                      default=None,
                      help="List of losses to track during evaluation")
    
    # positional embeddings
    args.add_argument("--scaling-factor", type=float, default=16.0) # TODO
    args.add_argument("--scaling-type", type=str, default="yarn") # TODO
    args.add_argument("--rope-theta", type=float, default=10000.0) # TODO
    args.add_argument("--original-max-position-embeddings", type=int)
    args.add_argument("--max-position-embeddings", type=int)
    # LR 
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--warmup-steps", type=int, default=10)
    args.add_argument("--warmup-steps-prefreeze", type=int, default=-1)
    args.add_argument("--lr-schedule", type=str, choices=["linear", "constant", "cosine"], default="linear")
    args.add_argument("--lr-schedule-prefreeze", type=str, choices=["linear", "constant", "cosine"], default="constant")
    
    # Model params
    args.add_argument("--model", type=str)
    args.add_argument("--original-model", type=str, default=None)
    # args.add_argument("--architecture", type=str,
    #                   choices=["llama", "mistral", "gemma"], default="llama")
    args.add_argument("--sliding-window-attention-schedule", type=str)  # TODO
    args.add_argument("--tokenizer-path", type=str)
    args.add_argument("--pre-tok-name", type=str, default="empty")
    
    # vocab extension params
    args.add_argument("--embedding-init-strategy", type=str,
                     choices=["default", "random", "clone", "mean", "zeros", "merge", None],
                     default=None,
                     help="Strategy for initializing new token embeddings")
    args.add_argument("--num-new-tokens", type=int,
                      default = 0,
                      help="Number of new tokens to add when extending.  Will check for compatibility with tokenizer.")


    # EVAL PARAMS
    args.add_argument("--eval-steps", type=int, default=100,
                     help="Number of steps between evaluations")
    args.add_argument("--eval-iters", type=int, default=100,
                     help="Number of iterations to run for evaluation")
    args.add_argument("--benchmark-steps", type=int, default=1000,
                     help="Number of steps between evaluations")
    args.add_argument("--run-lm-eval", action="store_true",
                     help="Run language model evaluation")
    args.add_argument("--eval-batch-size", type=int, default=2,
                     help="Batch size for evaluation")
    args.add_argument("--num-fewshot", type=int, default=-1,
                     help="Number of fewshot examples")
    args.add_argument("--limit", type=int, default=-1,
                     help="Number of samples to limit evaluation to")
    args.add_argument("--log-samples", action="store_true",
                     help="store actual samples from benchmark")
    args.add_argument("--benchmark-tasks", type=str, default="minerva_math",
                     help="Benchmark tasks to run")
    args.add_argument("--do-not-materialize-logits", action="store_true",
                     help="Whether to not materialize logits for additional time savings")
    args.add_argument("--task_list_split", type=str, default=None)

    args = args.parse_args()

    # some config validation
    args.wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else None
    # loss tracking

    allowed_loss_types = ["all", "translated", "new_tokens"]
    train_losses_to_track = args.train_losses_to_track.split(",") if args.train_losses_to_track else []
    for loss_type in train_losses_to_track:
        if loss_type not in allowed_loss_types:
            raise ValueError(f"Invalid train loss type: {loss_type}")
    args.train_losses_to_track = train_losses_to_track
        
    eval_losses_to_track = args.eval_losses_to_track.split(",") if args.eval_losses_to_track else ["all"]
    for loss_type in eval_losses_to_track:
        if loss_type not in allowed_loss_types:
            raise ValueError(f"Invalid eval loss type: {loss_type}")
        
    if args.task_name == "translation":
        if "translated" not in eval_losses_to_track:
            eval_losses_to_track.append("translated")
        # if "new_tokens" not in args.eval_losses_to_track:
        #     eval_losses_to_track.append("new_tokens")
    args.eval_losses_to_track = eval_losses_to_track

    if args.main_loss == None:
        args.main_loss = "all"
        if args.task_name == "SFT":
            args.main_loss = "all"
        elif args.task_name == "translation":
            args.main_loss = "translated"
        elif args.task_name == "mixed":
            args.main_loss = "mixed"
        else:
            raise ValueError(f"Invalid task name: {args.task_name}")
        
    if args.original_model is None:
        args.original_model = args.model
    
    return args


