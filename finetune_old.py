import argparse
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, DataCollatorWithPadding, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AutoModelForCausalLM, TrainingArguments, Trainer, TextStreamer, AutoTokenizer, AutoModelForSequenceClassification
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import torch.distributed as dist
import csv
import shutil
import math
from typing import Any

from accelerate.logging import get_logger
import logging

import psutil
from tokenize_simple import get_tokenized_data, flatten_genqa_conversations, my_tokenize, get_genqa_data, get_tokenizer

from liger_kernel.transformers import AutoLigerKernelForCausalLM

import gc


def setup_logging(log_level=logging.INFO):
    """Setup global logging configuration"""
    # Set up basic configuration first
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=log_level,
        force=True  # This ensures we override any existing configuration
    )
    
    # Get the logger
    ll = logging.getLevelName(log_level)
    logger = get_logger(__name__)
    # logger = logging.getLogger(__name__)  # if you want to see for all ranks
    logger.setLevel(ll)
    
    return logger

# logger = setup_logging(logging.DEBUG)
logger = setup_logging()

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def initialize_distributed():
    if not dist.is_initialized():
        logger.info("initializing process group")
        dist.init_process_group(backend='nccl')  # or 'gloo' depending on your setup


def log_loss(metrics_dict, output_dir, filename="loss_log.csv"):
    """Save metrics to a CSV file for tracking."""
    loss_file = os.path.join(output_dir, filename)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(loss_file)
    with open(loss_file, "a", newline="") as f:
        writer = csv.writer(f)
        # Only write header if file is new
        if not file_exists:
            writer.writerow(metrics_dict.keys())
        writer.writerow(metrics_dict.values())


def evaluate(model, eval_loader, accelerator, num_steps=100):
    model.eval()
            
    with torch.no_grad():
        losses = []
        tokens = []
        for i, batch in enumerate(eval_loader):
            if i >= num_steps:
                break
            num_items_in_batch = (batch["labels"].ne(-100)).sum().cpu().item()

            outputs = model(**batch, use_cache=False, num_items_in_batch=num_items_in_batch)
            loss_value = outputs.loss.item()
            
            gathered_metrics = accelerator.gather_for_metrics([loss_value, num_items_in_batch])
            loss_value = gathered_metrics[0::2]
            num_items_in_batch = gathered_metrics[1::2]
            
            losses.extend(loss_value)
            tokens.extend(num_items_in_batch)
            del outputs
                
        batch_sum_weighted_loss = 0
        batch_total_tokens = 0
        for loss, tok in zip(losses, tokens):
            batch_sum_weighted_loss += loss * tok
            batch_total_tokens += tok

        weighted_avg_loss = batch_sum_weighted_loss / batch_total_tokens
        avg_loss = sum(losses) / len(losses)

    return {"weighted_loss": weighted_avg_loss, "loss": avg_loss}


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

from dataclasses import dataclass
from typing import Dict, List, Union
import torch

@dataclass
class MyPaddingCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    tokenizer: Any
    max_length: int = None
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract all input_ids, attention_masks, and labels
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Convert to tensors if they aren't already
        if isinstance(input_ids[0], list):
            input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        if isinstance(attention_mask[0], list):
            attention_mask = [torch.tensor(x, dtype=torch.long) for x in attention_mask]
        if isinstance(labels[0], list):
            labels = [torch.tensor(x, dtype=torch.long) for x in labels]
        
        # Compute padding
        max_length_tokens = max(x.size(0) for x in input_ids)
        if self.max_length is None:
            max_length = max_length_tokens
        else:
            max_length = max(max_length_tokens, self.max_length)
        
        # Pad all tensors to max_length
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            # Calculate padding length
            pad_len = max_length - ids.size(0)
            
            if pad_len > 0:
                # Pad input_ids
                padded_input_ids.append(
                    torch.cat([ids, torch.ones(pad_len, dtype=torch.long) * self.tokenizer.pad_token_id])
                )
                # Pad attention_mask
                padded_attention_mask.append(
                    torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                )
                # Pad labels
                padded_labels.append(
                    torch.cat([lab, torch.ones(pad_len, dtype=torch.long) * -100])
                )
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
                padded_labels.append(lab)
        
        # Stack all tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }
        
        return batch

def move_optimizer_to_cpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):  # Check for tensor type
                state[k] = v.cpu()
                del v  # Explicitly delete GPU tensor
    torch.cuda.empty_cache()

def move_optimizer_to_gpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):  # Check for tensor type
                state[k] = v.cuda()
                del v  # Explicitly delete CPU tensor
    torch.cuda.empty_cache()

def log_memory_usage(step, phase, accelerator):
    """Log memory usage at various points in training"""
    if accelerator.is_local_main_process:
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"Step {step} - {phase} - "
                   f"GPU Memory Allocated: {gpu_memory_allocated:.2f}MB, "
                   f"Reserved: {gpu_memory_reserved:.2f}MB")

def find_optimal_batch_size(model, initial_batch_size, tokenizer, accelerator):
    """Dynamically find the largest batch size that fits in memory"""
    batch_size = initial_batch_size
    while batch_size > 1:
        try:
            # Create a sample batch
            sample_input = tokenizer("test" * 100, return_tensors="pt", padding=True)
            sample_input = {k: v.repeat(batch_size, 1).to(accelerator.device) for k, v in sample_input.items()}
            
            # Test forward and backward pass
            with torch.cuda.amp.autocast():
                outputs = model(**sample_input)
                loss = outputs.loss
                accelerator.backward(loss)
            
            del outputs, loss
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise e
    return 1

def create_memory_efficient_loader(dataset, batch_size, collate_fn, num_proc):
    """Create a memory-efficient data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,  # Reduce prefetching
        # persistent_workers=True,  # THIS throws an error
        num_workers=num_proc  # Adjust based on CPU cores
    )

def initialize_new_embeddings(
    base_embeddings: torch.nn.Parameter,
    num_new_tokens: int,
    init_strategy: str = "random",
    **kwargs
) -> torch.nn.Parameter:
    """
    Initialize embeddings for new tokens using different strategies.
    
    Args:
        base_embeddings: Original embedding weights
        num_new_tokens: Number of new tokens to add
        init_strategy: Strategy to use for initialization
            - "random": Standard normal initialization
            - "clone": Clone random existing embeddings
            - "mean": Initialize to mean of base embeddings
            - "zeros": Initialize to zeros
    """
    device = base_embeddings.device
    dtype = base_embeddings.dtype  # This will be bfloat16
    embed_dim = base_embeddings.shape[1]
    
    if init_strategy == "random":
        # Initialize directly with correct dtype
        new_embeddings = torch.empty(num_new_tokens, embed_dim, device=device, dtype=dtype)
        new_embeddings.normal_()  # In-place normal initialization
        logger.info(f"New embeddings dtype: {new_embeddings.dtype}, new embeddings shape: {new_embeddings.shape}, new embeddings: {new_embeddings}")
        # Scale appropriately
        std = base_embeddings.std().item()
        new_embeddings.mul_(std)
    
    elif init_strategy == "clone":
        # Randomly select tokens to clone
        indices = torch.randint(0, len(base_embeddings), (num_new_tokens,))
        new_embeddings = base_embeddings[indices].clone()
        # Add small random noise
        noise = torch.randn_like(new_embeddings) * 0.1
        new_embeddings += noise
    
    elif init_strategy == "mean":
        # Use mean of base embeddings
        mean_embedding = base_embeddings.mean(0, keepdim=True)
        new_embeddings = mean_embedding.repeat(num_new_tokens, 1)
        # Add small random noise
        noise = torch.randn_like(new_embeddings) * 0.1
        new_embeddings += noise
    
    elif init_strategy == "zeros":
        new_embeddings = torch.zeros(num_new_tokens, embed_dim, device=device, dtype=dtype)
    
    else:
        raise ValueError(f"Unknown initialization strategy: {init_strategy}")
    
    return new_embeddings

def extend_model_embeddings(model, tokenizer, init_strategy="random"):
    """Extend model embeddings to match new tokenizer vocabulary."""
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)
    
    if new_vocab_size <= base_vocab_size:
        # logger.warning("New vocabulary size is not larger than base vocabulary size!")
        return model
    
    num_new_tokens = new_vocab_size - base_vocab_size
    # logger.info(f"Extending vocabulary from {base_vocab_size} to {new_vocab_size} tokens")
    
    # Get the original embeddings
    old_embeddings = model.get_input_embeddings()
    old_weights = old_embeddings.weight.data
    
    # Initialize new embeddings
    new_token_embeddings = initialize_new_embeddings(
        old_weights, 
        num_new_tokens,
        init_strategy
    )
    
    # Create new embedding layer
    new_embeddings = torch.nn.Embedding(new_vocab_size, old_weights.shape[1])
    new_embeddings.to(dtype=old_weights.dtype, device=old_weights.device)
    
    # Copy old embeddings
    new_embeddings.weight.data[:base_vocab_size] = old_weights
    new_embeddings.weight.data[base_vocab_size:] = new_token_embeddings
    
    # Replace model embeddings
    model.set_input_embeddings(new_embeddings)
    
    # Resize output layer (lm_head) if it exists
    if hasattr(model, 'lm_head'):
        old_lm_head = model.lm_head
        new_lm_head = torch.nn.Linear(
            old_lm_head.in_features,
            new_vocab_size,
            bias=old_lm_head.bias is not None
        )
        
        # Copy old weights
        new_lm_head.weight.data[:base_vocab_size] = old_lm_head.weight.data
        new_lm_head.weight.data[base_vocab_size:] = new_token_embeddings  # Initialize with same values
        
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:base_vocab_size] = old_lm_head.bias.data
            new_lm_head.bias.data[base_vocab_size:] = 0  # Initialize new biases to 0
            
        model.lm_head = new_lm_head
    
    return model

def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        # mixed_precision="fp16",
        kwargs_handlers=[timeout],
        log_with="wandb" if args.wandb else None,
    )

    if accelerator.is_main_process:
        if args.wandb:
            import wandb
            wandb.login()
            wandb.init(project=args.wandb, name=f"Training Run {args.output_dir.split('/')[-1]}", config=vars(args))

    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "efficient-tokenization",
    )
    logger.info(f"Total GPUS: {accelerator.num_processes}")

    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,  # Disable KV cache during training
        # device_map="auto"  # Let accelerate handle device mapping
    )

    # # Get model
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     attn_implementation="sdpa",  # Enables Flash Attention v2
    # )


    if hasattr(model, "enable_input_require_grads"):  
        model.enable_input_require_grads()  # ✅ Ensures proper gradient tracking
    model.gradient_checkpointing_enable()
    logger.info("Loading model and tokenizer...")

    gradient_accumulation_steps = args.gradient_accumulate_every
    num_epochs = args.num_epochs

    if args.tokenizer_path is not None:
        # load tokenizer from vocab file
        base_tokenizer = AutoTokenizer.from_pretrained(args.model)
        vocab_file_path = args.tokenizer_path
        tokenizer = get_tokenizer(vocab_file_path, old_tokenizer=base_tokenizer)
        logger.info(f"Loaded tokenizer from vocab file: {vocab_file_path}")

    else:
        # get original_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info(f"Loaded tokenizer from model: {args.model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer vocab size: {len(tokenizer)}, pad token: {tokenizer.pad_token}, eos token: {tokenizer.eos_token}, pad_token_id: {tokenizer.pad_token_id}")

    if args.embedding_init_strategy is not None:
        # Extend model embeddings
        logger.info(f"Extending model embeddings with strategy: {args.embedding_init_strategy}")
        model = extend_model_embeddings(
            model, 
            tokenizer, 
            init_strategy=args.embedding_init_strategy
        )

    ds = load_from_disk(args.dataset)
    # Split the dataset into train (90%) and validation (10%)
    ds = ds.train_test_split(test_size=0.1)

    data_collator = MyPaddingCollator(
        tokenizer=tokenizer,
        max_length=args.max_length if hasattr(args, 'max_length') else None
    )

    train_loader = create_memory_efficient_loader(
        ds["train"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_proc=args.num_proc
    )
    train_samples = len(ds["train"])
    train_batches = len(train_loader)
    train_loader_token_count = sum(ds["train"]["num_tokens"])

    eval_loader = create_memory_efficient_loader(
        ds["test"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_proc=args.num_proc
    )
    eval_samples = len(ds["test"])
    eval_batches = len(eval_loader)
    eval_loader_token_count = sum(ds["test"]["num_tokens"])

    logger.info("Loaded data into dataset")

    # if args.lora:
    #     from peft import get_peft_model, LoraConfig, TaskType
    #     target_modules = find_all_linear_names(model)
    #     my_logger(accelerator, f"LoRA target modules: {target_modules}")
    #     peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
    #                              r=16, lora_alpha=64, lora_dropout=0.05, target_modules=target_modules)
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    # if args.deepspeed:
    #     from accelerate.utils import DummyOptim, DummyScheduler
    #     optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    #     scheduler = DummyScheduler(
    #         optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
    #     model, optim, train_loader, scheduler = accelerator.prepare(
    #         model, optim, train_loader, scheduler
    #     )
    # else:
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    logger.info(f"Initial learning rate from optimizer: {optim.param_groups[0]['lr']}")



    if args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
    else:
        max_train_steps = train_batches//(gradient_accumulation_steps * accelerator.num_processes)

    # these calculations are done AFTER the accelerator is initialized
    num_batches_per_device_per_epoch = len(train_loader)
    grad_updates_per_device_per_epoch = math.ceil(num_batches_per_device_per_epoch / gradient_accumulation_steps)

    epoch_remainder_on_device = num_batches_per_device_per_epoch % gradient_accumulation_steps
    epoch_remainder_on_device = epoch_remainder_on_device if epoch_remainder_on_device != 0 else gradient_accumulation_steps

    if num_epochs > 0:
        total_gradient_updates = min(max_train_steps, grad_updates_per_device_per_epoch * num_epochs * accelerator.num_processes)
        total_gradient_updates_per_device = min(math.ceil(max_train_steps // accelerator.num_processes), grad_updates_per_device_per_epoch * num_epochs)
    else:
        total_gradient_updates = max_train_steps
        total_gradient_updates_per_device = math.ceil(max_train_steps // accelerator.num_processes)


    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optim, 
            num_training_steps=max_train_steps,
            num_warmup_steps=args.warmup_steps
        )
    elif args.lr_schedule == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optim, 
            num_warmup_steps=args.warmup_steps
        )

    logger.info("Preparing artifacts")

    # prepare artifacts - accelerator handles device placement and dataloader splitting
    model, optim = accelerator.prepare(model, optim)
    train_loader = accelerator.prepare_data_loader(train_loader, device_placement=True)
    eval_loader = accelerator.prepare_data_loader(eval_loader, device_placement=True)
    training_iterator = iter(train_loader)

    split_scheduler = False
    if split_scheduler:
        scheduler = accelerator.prepare(scheduler)

    # Batch size stuff
    accelerator.register_for_checkpointing(scheduler)
    total_batch_size = (
        args.batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    def get_next_batch(training_iterator, train_loader, epoch):
        try:
            batch = next(training_iterator)
            # Immediately move to CPU if not needed
            batch = {k: v.cpu() for k, v in batch.items()}
            return batch, training_iterator, epoch
        except StopIteration:
            # End of epoch reached, create new iterator
            epoch += 1
            logger.info(f"Starting epoch {epoch}")
            training_iterator = iter(train_loader)
            batch = next(training_iterator)
            batch = {k: v.cpu() for k, v in batch.items()}
            return batch, training_iterator, epoch

    logger.info(f"max train steps {max_train_steps} ({math.ceil(max_train_steps / accelerator.num_processes)} per device)")
    logger.info(f"total gradient updates {total_gradient_updates} ({total_gradient_updates_per_device} per device)")
    logger.info(f"grad updates per epoch {grad_updates_per_device_per_epoch * accelerator.num_processes} ({grad_updates_per_device_per_epoch} per device)")
    logger.info(f"train samples: {train_samples}, batches: {train_batches}, and tokens: {train_loader_token_count}")
    logger.info(f"train samples per device: {len(train_loader) * args.batch_size}, batches per device: {len(train_loader)}, and tokens (approx): {train_loader_token_count / accelerator.num_processes}")
    logger.info(f"eval samples: {eval_samples}, batches: {eval_batches}, and tokens: {eval_loader_token_count}")
    logger.info(f"eval samples per device: {len(eval_loader) * args.batch_size}, batches per device: {len(eval_loader)}, and tokens (approx): {eval_loader_token_count / accelerator.num_processes}")
    logger.info(f"gradient steps {gradient_accumulation_steps}")
    logger.info(f"per device batch size {args.batch_size}")
    logger.info(f"num_processes {accelerator.num_processes}")
    logger.info(f"effective batch size {total_batch_size}")
    logger.info(f"learning rate {args.learning_rate}, warmup steps {args.warmup_steps}, schedule_max_steps {max_train_steps}, split_scheduler {split_scheduler}")
    logger.info(f"checkpointing steps {args.checkpointing_steps}")

    # if not args.lora:
    #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #         model.module.gradient_checkpointing_enable()
    #     else:
    #         model.gradient_checkpointing_enable()

    # # checkpoing resume
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
    #         logger.info(f"Resuming from checkpoint {args.resume_from_checkpoint}")
    #         accelerator.load_state(args.resume_from_checkpoint)
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     training_difference = os.path.splitext(path)[0]

    #     resume_step = (
    #         int(training_difference.replace("step_", ""))
    #     )


    # completed_steps = 0
    # if args.resume_from_checkpoint and resume_step is not None:
    #      # TODO this does not work yet
    #     train_loader = accelerator.skip_first_batches(train_loader, resume_step)
    #     completed_steps += resume_step
    #     progress_bar.update(resume_step)
    #     logger.info(f"Resuming training from step {resume_step}")

    # samples tracking
    epoch = 0
    epoch_step = -1
    update_step = -1
    total_batched_samples = 0

    # loss tracking
    cumulative_batch_counter = 0  # number of batches processed since last accumulation
    cumulative_token_counter = 0  # all tokens processed across all processes

    can_train = True
    if args.save_only:
        can_train = False

    logger.info(f"Training for {max_train_steps} steps")
    use_progress_bar = True
    progress_bar = tqdm(
        range(total_gradient_updates),
        disable=(not use_progress_bar) or (not accelerator.is_local_main_process),  # turning off progress bar
        # disable=not accelerator.is_local_main_process,
    )
    largest_batch_per_device = 0
    # for update_step in range(total_gradient_updates_per_device):
    while update_step < total_gradient_updates:
        epoch_step += 1
        update_step += 1
        # logger.info(f"Starting epoch {epoch}")
        logger.debug(f"Starting step {update_step} of {total_gradient_updates}")
        
        # More aggressive memory clearing between steps
        accelerator.clear()
        torch.cuda.empty_cache()
        gc.collect()  # Add garbage collection
        
        if not can_train:
            break

        mem_before = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"Memory before loading batches - Device {accelerator.process_index}: {mem_before:.2f} MB", main_process_only=False)

        model.train()
        # TODO add time of each iteration, and memory usage
    
        loss_log = None
        
        # try this
        batch_samples = []
        
        num_batches_in_step = gradient_accumulation_steps if epoch_step != (grad_updates_per_device_per_epoch - 1) else epoch_remainder_on_device
        logger.debug(f"gradient accumulation steps: {gradient_accumulation_steps} total gradient updates: {total_gradient_updates}, remainder: {epoch_remainder_on_device}, num_batches_in_step {num_batches_in_step}")
        for _ in range(num_batches_in_step):
            batch, training_iterator, epoch = get_next_batch(training_iterator, train_loader, epoch)
            this_batch = {k: v.cpu() for k, v in batch.items()}
            batch_samples.append(this_batch)
            del batch  # Explicitly delete the original batch

        # get local num items in batch
        device_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"Step {update_step} - Device {accelerator.process_index} - device memory usage {device_memory_usage} MB, largest_batch_per_device {largest_batch_per_device}", main_process_only=False)

        accumulated_losses = []
        accumulated_token_counts = []
        accumulation_batch_counter = 0
        # logger.debug(f"batch_samples length: {len(batch_samples)}, batch_samples[0]: {batch_samples[0]}, context len: {batch_samples[0]['input_ids'].shape[1]}", main_process_only=True)
        
        for i, batch in enumerate(batch_samples):
            total_batched_samples += 1
            # Move batch to device just before use
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            num_items_in_batch = (batch["labels"].ne(-100)).sum().cpu()
            mem_before_forward = torch.cuda.memory_allocated() / 1024**2
            largest_batch_per_device = max(largest_batch_per_device, batch['input_ids'].numel())

            logger.debug(f"Memory before forward pass opt_step: {update_step} accum_step:{i} - Device {accelerator.process_index}: {mem_before_forward:.2f} MB, "
                f"num_items_in_batch {num_items_in_batch} samples: {batch['input_ids'].shape[0]}, "
                f"{[(batch['labels'][x].ne(-100)).sum().item() for x in range(batch['input_ids'].shape[0])]}, largest_batch_per_device {largest_batch_per_device}",
                main_process_only=False
                )


            # IMPORTANT:
            # the losses will be separate for each batch, process
            # but the gradients accumulate across all processes.
            # therefore you dont need to only do grad.step() on the main process/sync step because it knows   
            with accelerator.accumulate(model):  # Use accelerator's context manager
                # 1. Forward pass
                outputs = model(**batch, use_cache=False, num_items_in_batch=num_items_in_batch)
                loss = outputs.loss
                # loss = (loss * gradient_accumulation_steps * accelerator.num_processes)

                mem_before_backward = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory before backwards pass opt_step: {update_step} accum_step:{i} - Device {accelerator.process_index}: {mem_before_backward:.2f} MB", main_process_only=False)

                accelerator.backward(loss)
                # Immediate cleanup after backward pass

                accumulated_losses.append(loss.item())  # Move loss to CPU immediately
                accumulated_token_counts.append(num_items_in_batch.item())
                accumulation_batch_counter += 1

                # DO SIMPLE AVERAGE OVER ACCUMULATION STEPS

                del outputs
                del loss
                torch.cuda.empty_cache()  # Consider moving this outside the loop
                
                # TODO check if this is correct
                if args.grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.grad_norm)


                optim.step()
                # TODO look at single neuron and gradient
                if accelerator.sync_gradients:
                    logger.debug(f"Device {accelerator.process_index} - Step {update_step}, {accumulation_batch_counter}: {scheduler.get_last_lr()}")
                    scheduler.step()
                    
                optim.zero_grad()
                # logger.info(f"losses: {accumulated_losses} accumulated_token_counts: {accumulated_token_counts}  accumulated_batch_counter: {accumulation_batch_counter} grad_norm: {grad_norm} learning rate: {scheduler.get_last_lr()[0]}", main_process_only=False)
    
        accelerator.clear()
        torch.cuda.empty_cache()
                
        if accelerator.sync_gradients:
            logger.debug(f"SYNC POINT - Steps {update_step}, "
                        f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}, "
                        f"Gradient state steps: {accelerator.gradient_state.num_steps}, "
                        f"Our counter: {accumulation_batch_counter}")

            # All processes must participate in gather
            gathered_metrics = accelerator.gather_for_metrics([
                accumulated_losses,
                accumulated_token_counts,
                accumulation_batch_counter
            ])

            if accelerator.is_local_main_process:
                # Regroup the metrics by type
                gathered_losses = [item for sublist in gathered_metrics[0::3] for item in sublist]
                gathered_tokens = [item for sublist in gathered_metrics[1::3] for item in sublist]
                gathered_batch_counter = sum(gathered_metrics[2::3])
                
                # TODO confirm weighted loss is correct
                batch_sum_weighted_loss = 0
                batch_total_tokens = 0
                for loss, tokens in zip(gathered_losses, gathered_tokens):
                    batch_sum_weighted_loss += loss * tokens
                    batch_total_tokens += tokens

                avg_loss = batch_sum_weighted_loss / batch_total_tokens
                cumulative_token_counter += batch_total_tokens
                cumulative_batch_counter += gathered_batch_counter
            
                # Create metrics dict
                metrics_dict = {
                    "weighted_loss": avg_loss,
                    "token_counter": batch_total_tokens,
                    "cum_token_counter": cumulative_token_counter,
                    "batch_counter": gathered_batch_counter,
                    "cum_batch_counter": cumulative_batch_counter,
                    "lr": scheduler.get_last_lr()[0]
                }
                metrics_dict = {f"train_{k}": v for k, v in metrics_dict.items()}
                metrics_dict["step"] = update_step
                loss_log = {
                    "loss": avg_loss,
                }
            
                log_loss(metrics_dict, args.output_dir)
                if args.wandb:
                    wandb.log(metrics_dict)
            
                # only update progress bar on main process also only do checkpointing on main process
                progress_bar.update(1)
                if loss_log is not None:
                    if use_progress_bar:
                        progress_bar.set_postfix(loss_log)  # This will now show both raw loss and moving average
                    else:
                        logger.info(metrics_dict)

                # checkpointing
                if isinstance(args.checkpointing_steps, int) and update_step > 0:
                    if update_step % args.checkpointing_steps == 0:
                        logger.info(f"Checkpointing step {update_step}")
                        output_dir = f"step_{update_step}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            f"{args.output_dir}/{output_dir}",
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            # state_dict=state_dict,
                        )

            del gathered_metrics
    
            if update_step % args.eval_steps == 0:
                logger.debug(f"EVAL:Step {update_step}: Starting evaluation")
                move_optimizer_to_cpu(optim)
                torch.cuda.empty_cache()
                eval_metrics_dict = evaluate(model, eval_loader, accelerator, args.eval_iters)
                move_optimizer_to_gpu(optim)
                logger.debug(f"EVAL:Step {update_step}: Eval Loss = {eval_metrics_dict['loss']:.4f}, Weighted Eval Loss = {eval_metrics_dict['weighted_loss']:.4f}")

                eval_metrics_dict = {f"eval_{k}": v for k, v in eval_metrics_dict.items()}
                eval_metrics_dict["step"] = update_step

                if accelerator.is_main_process:
                    # Use the same dict for both logging and wandb
                    log_loss(eval_metrics_dict, args.output_dir, filename="loss_log_eval.csv")
                    if args.wandb:
                        wandb.log(eval_metrics_dict)
                
                    logger.debug(eval_metrics_dict)
                #TODO  add early stopping
    
    logger.info(f"Training Finished")

    accelerator.clear()  # This clears the accelerator's internal state

    torch.cuda.empty_cache()
    
    # Make sure all processes are synced
    accelerator.wait_for_everyone()    

    if args.output_dir is not None:
        logger.info(f"Preparing to save model to {args.output_dir}")
        
        if not check_disk_space(args.output_dir, required_space_gb=20):
            logger.info("Aborting save due to insufficient disk space")
            return
        
        if torch.distributed.is_initialized():
            accelerator.wait_for_everyone()

        # if args.deepspeed:
        #     state_dict = accelerator.get_state_dict(model)
        # else:
        #     full_state_dict_config = FullStateDictConfig(
        #         offload_to_cpu=True, rank0_only=True)
        #     # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        #     #     state_dict = accelerator.get_state_dict(model, unwrap=False)
        #     state_dict = FSDP.get_state_dict(model, state_dict_config=full_state_dict_config)
        #     # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        #     #     state_dict = model.state_dict()

        save_model = True
        if save_model:
            output_dir = f"final_model"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                # state_dict=state_dict,
            )

        move_optimizer_to_cpu(optim)
        torch.cuda.empty_cache()
        eval_metrics_dict = evaluate(model, eval_loader, accelerator, args.eval_iters)
        move_optimizer_to_gpu(optim)
        logger.debug(f"EVAL FINAL: Eval Loss = {eval_metrics_dict['loss']:.4f}, Weighted Eval Loss = {eval_metrics_dict['weighted_loss']:.4f}")

        eval_metrics_dict = {f"eval_{k}": v for k, v in eval_metrics_dict.items()}
        eval_metrics_dict["step"] = update_step


        if accelerator.is_main_process:
            # Use the same dict for both logging and wandb
            log_loss(eval_metrics_dict, args.output_dir, filename="loss_log_eval.csv")
            if args.wandb:
                wandb.log(eval_metrics_dict)
            
        logger.info(f"Loss log saved to {args.output_dir}/loss_log.csv")
        logger.info(f"Saving Finished")

    accelerator.end_training()


if __name__ == "__main__":
    try:
        threads = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
    except:
        threads = os.cpu_count()

    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--cpu-batch-size", type=int, default=1000)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=-1)
    args.add_argument("--num-epochs", type=int, default=-1)
    args.add_argument("--warmup-steps", type=int, default=10)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--grad-norm", type=float, default=1.0, 
                     help="Max norm for gradient clipping. Set to None to disable.")
    args.add_argument("--lora", action="store_true")
    args.add_argument("--model", type=str,
                      default="meta-llama/Llama-3.2-1B")
    args.add_argument("--scaling-factor", type=float, default=16.0)
    args.add_argument("--scaling-type", type=str, default="yarn")
    args.add_argument("--rope-theta", type=float, default=10000.0)
    args.add_argument("--truncate", type=int)
    args.add_argument("--dataset", type=str,
                      default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    args.add_argument("--deepspeed", action="store_true")
    
    args.add_argument("--num-proc", type=int, default=threads)
    # args.add_argument("--architecture", type=str,
    #                   choices=["llama", "mistral", "gemma"], default="llama")
    args.add_argument("--max-position-embeddings", type=int)
    args.add_argument("--sliding-window-attention-schedule", type=str)
    args.add_argument("--lr-schedule", type=str,
                      choices=["linear", "constant"], default="linear")
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--save-dataset-path", type=str)
    args.add_argument("--tokenizer-path", type=str)
    args.add_argument("--force-tokenize", action="store_true")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--original-max-position-embeddings", type=int)
    args.add_argument("--eval-steps", type=int, default=1000,
                     help="Number of steps between evaluations")
    args.add_argument("--eval-iters", type=int, default=100,
                     help="Number of iterations to run for evaluation")
    args.add_argument("--memory-efficient", action="store_true", 
                     help="Enable memory efficient training mode")
    args.add_argument("--auto-batch-size", action="store_true",
                     help="Automatically find optimal batch size")
    args.add_argument("--cpu-offload", action="store_true",
                     help="Enable CPU offloading for optimizer states")
    args.add_argument("--embedding-init-strategy", type=str,
                     choices=["random", "clone", "mean", "zeros"],
                     default="random",
                     help="Strategy for initializing new token embeddings")
    main(args.parse_args())

    # TODO consider training params:
    # - grad norm
    # - lora
