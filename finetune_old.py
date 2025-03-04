import argparse
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import set_seed, DataCollatorWithPadding, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AutoModelForCausalLM, TrainingArguments, Trainer, TextStreamer, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import datasets
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import torch.distributed as dist
import csv
import shutil
import math
from typing import Any, Dict

from accelerate.logging import get_logger
import logging
import json

import psutil
from tokenize_simple import get_tokenized_data, flatten_genqa_conversations, my_tokenize, get_genqa_data, get_tokenizer
from extend_embeddings import extend_model_embeddings, initialize_new_embeddings, get_new_embedding_params, get_new_embeddings_grads

from lm_eval import evaluator, tasks, utils, models
from lm_eval.tasks import TaskManager


from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerCrossEntropyLoss

import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def is_distributed():
    return torch.distributed.is_initialized() if torch.distributed.is_available() else False


def setup_logging(log_level=logging.INFO):
    """Setup global logging configuration"""
    # Set up basic configuration first
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=log_level,
        force=True  # This ensures we override any existing configuration
    )
    if AcceleratorState().initialized:
        ll = logging.getLevelName(log_level)
        logger = get_logger(__name__)
        logger.setLevel(ll)
    else:
        logger = logging.getLogger(__name__)  # if you want to see for all ranks
    
    return logger

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

def benchmark(model, config):
    """
    Run minerva math benchmark evaluation
    """
    model_args_dict = {
        "pretrained": model,  # This will be your model object directly
        # "tokenizer": config["tokenizer_path"],  # This will be your tokenizer object directly
        "tokenizer": config["tokenizer"],  # This will be your tokenizer object directly
        # "old_tokenizer": config["base_tokenizer"],  # This will be your tokenizer object directly
        # "pre_tok_name": config["pre_tok_name"],
        # "parallelize": True,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 3,
        "trust_remote_code": True
    }
    if model_args_dict.get("trust_remote_code", False):
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        
    model_args = ",".join([f"{k}={v}" for k, v in model_args_dict.items()])

    LM_model = models.huggingface.HFLM(**model_args_dict)

    # hf_hub_log_args = ""
    # evaluation_tracker_args = simple_parse_args_string(hf_hub_log_args)
    # evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)
    # apply_chat_template = False
    # fewshot_as_multiturn = False
    # verbosity = "INFO"
    verbosity = "CRITICAL"
    # predict_only = False
    # default_seed_string = "0,1234,1234,1234"

    task_names = config["benchmark_tasks"]
    task_manager = TaskManager()
    task_list = task_names.split(",")
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
        if os.path.isfile(task):
            config = utils.load_yaml_config(task)
            task_names.append(config)
    task_missing = [
        task for task in task_list if task not in task_names and "*" not in task
    ]  # we don't want errors if a wildcard ("*") task name was used
    if task_missing:
        missing = ", ".join(task_missing)
        # eval_logger.error(
        #     f"Tasks were not found: {missing}\n"
        #     f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
        # )
        raise ValueError(
            f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
        )


    results = evaluator.simple_evaluate(
        model=LM_model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=config["num_fewshot"],
        batch_size=config["batch_size"],
        # max_batch_size=max_batch_size,
        # device=device,
        # use_cache=config.get("use_cache", None),
        limit=config["limit"],
        # check_integrity=check_integrity,
        # write_out=write_out,
        log_samples=config["log_samples"],
        # evaluation_tracker=evaluation_tracker,
        # system_instruction=system_instruction,
        # apply_chat_template=apply_chat_template,
        # fewshot_as_multiturn=fewshot_as_multiturn,
        # gen_kwargs=gen_kwargs,
        task_manager=task_manager,
        verbosity=verbosity,
        # predict_only=predict_only,
        # random_seed=seed[0],
        # numpy_random_seed=seed[1],
        # torch_random_seed=seed[2],
        # fewshot_random_seed=seed[3],
        # confirm_run_unsafe_code=confirm_run_unsafe_code,
        # **request_caching_args,
    )

    output_results = {}
    for task, info in results["results"].items():
        output_results[f"{task}_acc"] = info['exact_match,none']

    return output_results

def run_evaluation(model, eval_loader, accelerator, num_steps):
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


def evaluate(model, eval_loader, accelerator, num_steps=100, eval_config: Dict = None):
    model.eval()
    if eval_config is None:
        eval_config = {}
    results = {}
    if eval_config.get("run_lm_eval", False) and accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        lm_eval_results = benchmark(unwrapped_model, eval_config)
        # print(json.dumps(lm_eval_results, indent=4))
        results["lm_eval"] = lm_eval_results
        # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    accelerator.wait_for_everyone()
    
    eval_loop_results = run_evaluation(model, eval_loader, accelerator, num_steps)

    results.update(eval_loop_results)
    return results


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
import torch.nn as nn

# overwrite the forward method of the model
def my_custom_forward(self, input_ids, attention_mask=None, labels=None, loss_mask=None, **kwargs):
    # 1) Call original forward, which we stored as _old_forward
    outputs = self._old_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,          # Force returning a ModelOutput
        output_attentions=False,   # Optional
        output_hidden_states=False, # Optional
        **kwargs
    )
    # outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    print(outputs)
    if labels is not None and loss_mask is not None:
        # Get the original loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_loss_mask = loss_mask[..., 1:].contiguous()
        
        # Calculate loss only where loss_mask is 1
        loss_fct = nn.functional.cross_entropy(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape)
        
        # Apply mask and average
        masked_loss = (loss * shift_loss_mask).sum() / shift_loss_mask.sum()
        outputs.loss = masked_loss

        # Calculate loss for new tokens if applicable
        if "new_token_start_index" in kwargs:
            new_token_start_index = kwargs["new_token_start_index"]
            new_token_mask = (shift_labels > new_token_start_index).float()
            if new_token_mask.sum() > 0:
                masked_loss_new_tokens = (loss * new_token_mask).sum() / new_token_mask.sum()
            else:
                masked_loss_new_tokens = torch.tensor(0.0, device=loss.device)
            outputs.loss_new_tokens = masked_loss_new_tokens
        else:
            outputs.loss_new_tokens = torch.tensor(0.0, device=loss.device)
            
    return outputs

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


@dataclass
class MyPaddingCollatorWithLossMask:
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
        loss_mask = [f["loss_mask"] for f in features]

        # Convert to tensors if they aren't already
        if isinstance(input_ids[0], list):
            input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        if isinstance(attention_mask[0], list):
            attention_mask = [torch.tensor(x, dtype=torch.long) for x in attention_mask]
        if isinstance(labels[0], list):
            labels = [torch.tensor(x, dtype=torch.long) for x in labels]
        if isinstance(loss_mask[0], list):
            loss_mask = [torch.tensor(x, dtype=torch.long) for x in loss_mask]

        loss_mask_list = []
        # only calculate loss on the tokens that are not masked
        for label, loss_mask_item in zip(labels, loss_mask):
            label[loss_mask_item==0] = -100
            loss_mask_list.append(label)

        loss_mask = loss_mask_list
        del loss_mask_list

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
        padded_loss_mask = []

        for ids, attn_mask, lab, loss_mask_item in zip(input_ids, attention_mask, labels, loss_mask):
            # Calculate padding length
            pad_len = max_length - ids.size(0)
            
            if pad_len > 0:
                # Pad input_ids
                padded_input_ids.append(
                    torch.cat([ids, torch.ones(pad_len, dtype=torch.long) * self.tokenizer.pad_token_id])
                )
                # Pad attention_mask
                padded_attention_mask.append(
                    torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
                )
                # Pad labels
                padded_labels.append(
                    torch.cat([lab, torch.ones(pad_len, dtype=torch.long) * -100])
                )
                # Pad loss_mask
                padded_loss_mask.append(
                    torch.cat([loss_mask_item, torch.zeros(pad_len, dtype=torch.long)])
                )
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(attn_mask)
                padded_labels.append(lab)
                padded_loss_mask.append(loss_mask_item)
                
        # Stack all tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "loss_mask": torch.stack(padded_loss_mask)
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


def calculate_grad_norm(parameters, norm_type=2.0, error_if_nonfinite=False, foreach=None, is_grad=False):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    if not is_grad:
        grads = [p.grad for p in parameters if p.grad is not None]
    else:
        grads = parameters
    # print(f"parameters: {[p.shape for p in parameters]}, grads:{[g.shape for g in grads]}, {grads}")
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    return total_norm


def freeze_model_except_embeddings(model, freeze_output_embeddings=True):
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze input embeddings
    if hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings().weight.requires_grad = True
    
    # Optionally unfreeze output embeddings if they're not tied
    if not freeze_output_embeddings and hasattr(model, 'get_output_embeddings'):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight.requires_grad = True
    
    # Print stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

def main(args):

    global logger  # Update the logger once accelerator is initialized

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        # mixed_precision="fp16",
        kwargs_handlers=[timeout],
        log_with="wandb" if args.wandb else None,
    )

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        total_batch_size = args.batch_size * args.gradient_accumulate_every * accelerator.num_processes
        if args.embedding_init_strategy is not None:
            output_dir = f"output/{args.model.split('/')[-1]}-task_{args.task_name}-finetuning_mode_{args.finetuning_mode}-batch{total_batch_size}-extend_{args.embedding_init_strategy}"
        else:
            output_dir = f"output/{args.model.split('/')[-1]}-task_{args.task_name}-finetuning_mode_{args.finetuning_mode}-batch{total_batch_size}"


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(logging.DEBUG)
    # logger = setup_logging()

    if accelerator.is_main_process:
        if args.wandb:
            import wandb
            tags = args.wandb_tags.split(",") if args.wandb_tags else None
            tags = tags if len(tags) > 0 else None
            logger.info(f"Logging to wandb with tags: {tags}")
            wandb.login()
            wandb.init(project=args.wandb,
                       tags=tags,
                       name=f"Training Run {output_dir.split('/')[-1]}", 
                       config=vars(args)
                       )

    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "efficient-tokenization",
    )
    logger.info(f"Total GPUS: {accelerator.num_processes}")

    save_model_type = args.save_model
    if args.dry_run:
        save_model_type = None

    # Load config
    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cross_entropy=True,
        fused_linear_cross_entropy=False,
        # config=config,
        # use_cache=False,  # Disable KV cache during training
    )
    logger.debug(f"Overwriting forward method of model")
    # model._old_forward = model.forward
    # model.forward = my_custom_forward.__get__(model, model.__class__)

    # TODO: implement finetuning modes
    finetuning_mode = args.finetuning_mode
    if finetuning_mode == "new_tokens_only":
        raise NotImplementedError("New tokens only finetuning is not implemented yet")
    elif finetuning_mode == "embeddings":
        raise NotImplementedError("Embeddings finetuning is not implemented yet")
    elif finetuning_mode == "full":
        pass
    else:
        raise ValueError(f"Invalid finetuning mode: {finetuning_mode}")

    if hasattr(model, "enable_input_require_grads"):  
        model.enable_input_require_grads()

    model.gradient_checkpointing_enable()
    logger.info("Loading model and tokenizer...")

    gradient_accumulation_steps = args.gradient_accumulate_every
    num_epochs = args.num_epochs

    base_tokenizer = None
    vocab_file_path = None
    pre_tok_name = None

    if args.tokenizer_path is not None:
        # load tokenizer from vocab file
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            logger.info(f"Loaded HF tokenizer from vocab file: {args.tokenizer_path}")
        except:
            base_tokenizer = AutoTokenizer.from_pretrained(args.model)
            vocab_file_path = args.tokenizer_path
            pre_tok_name = args.pre_tok_name
            tokenizer = get_tokenizer(vocab_file_path, pre_tok_name=pre_tok_name, old_tokenizer=base_tokenizer)
            logger.info(f"Loaded tokenizer from .model file: {vocab_file_path}")
    else:
        # get original_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info(f"Loaded tokenizer from model: {args.model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    num_new_tokens = 0
    original_vocab_size = model.config.vocab_size
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}, model vocab size: {original_vocab_size}")
    
    num_new_tokens = len(tokenizer) - model.config.vocab_size
    if args.embedding_init_strategy is not None and num_new_tokens > 0:
        # Extend model embeddings
        logger.info(f"Extending model embeddings with strategy: {args.embedding_init_strategy} and adding {num_new_tokens} new tokens")
        model = extend_model_embeddings(
            model, 
            num_new_tokens, 
            init_strategy=args.embedding_init_strategy,
            tokenizer=tokenizer
        )
    
    embedding_size_in = model.get_input_embeddings().weight.shape[0]
    embedding_size_out = model.get_output_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer.get_vocab())
    if embedding_size_in != new_vocab_size or embedding_size_out != new_vocab_size:
        raise ValueError(f"Embedding size {embedding_size_in}/{embedding_size_out} (input/output) does not match vocab size {new_vocab_size}")

    # After model loading
    logger.info(f"Model config: max_position_embeddings={model.config.max_position_embeddings}, "
               f"vocab_size={model.config.vocab_size}")

    if hasattr(model.config, "rope_scaling"):
        logger.info(f"RoPE scaling config: {model.config.rope_scaling}")

    # TODO: implement task 
    task_name = args.task_name
    if task_name == "SFT":
        data_collator = MyPaddingCollator(
            tokenizer=tokenizer,
            max_length=args.max_length if hasattr(args, 'max_length') else None
        )
    elif task_name == "translation":
        data_collator = MyPaddingCollatorWithLossMask(
            tokenizer=tokenizer,
            max_length=args.max_length if hasattr(args, 'max_length') else None
        )
    elif task_name == "mixed":
        raise NotImplementedError("Mixed task is not implemented yet")
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    
    ds = load_from_disk(args.dataset)
    # Split the dataset into train (90%) and validation (10%)
    ds = ds.train_test_split(test_size=0.1)

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
        batch_size=args.eval_batch_size,  # TODO make sure this works
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
    logger.info(f"eval samples per device: {len(eval_loader) * args.eval_batch_size}, batches per device: {len(eval_loader)}, and tokens (approx): {eval_loader_token_count / accelerator.num_processes}")
    logger.info(f"gradient steps {gradient_accumulation_steps}")
    logger.info(f"per device batch size {args.batch_size}, eval batch size {args.eval_batch_size}")
    logger.info(f"accelerator distributed type {accelerator.distributed_type}, num_processes {accelerator.num_processes}")
    logger.info(f"effective batch size {total_batch_size}")
    logger.info(f"learning rate {args.learning_rate}, warmup steps {args.warmup_steps}, schedule_max_steps {max_train_steps}, split_scheduler {split_scheduler}")
    logger.info(f"checkpointing steps {args.checkpointing_steps}")
    logger.info(f"")

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

    eval_config = {
        "run_lm_eval": args.run_lm_eval,
        "batch_size": args.eval_batch_size,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "log_samples": args.log_samples,
        "benchmark_tasks": args.benchmark_tasks,
    }
    eval_config["tokenizer"] = tokenizer
    if base_tokenizer is not None:
        eval_config["base_tokenizer"] = base_tokenizer
    if vocab_file_path is not None:
        eval_config["tokenizer_path"] = vocab_file_path
    if pre_tok_name is not None:
        eval_config["pre_tok_name"] = pre_tok_name


    largest_batch_per_device = 0
    while update_step < total_gradient_updates - 1:
        epoch_step += 1
        update_step += 1
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
        grad_norm = None
        new_embeddings_grad_norm = None
        
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
        accumulated_losses_new_tokens = []
        accumulated_token_counts = []
        accumulation_batch_counter = 0
        new_token_counter = []
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
            # Before model forward pass
            # Check if any new tokens are present in the batch
            new_tokens_mask = batch['input_ids'] >= original_vocab_size
            num_new_tokens_in_batch = new_tokens_mask.sum().item()
            max_token_id = batch['input_ids'].max().item()
            vocab_size = model.module.config.vocab_size
            if max_token_id >= vocab_size:
                raise ValueError(
                    f"Token ID {max_token_id} found in batch, but vocab size is only {vocab_size}. "
                    f"This will cause CUDA indexing errors."
                )

            # IMPORTANT:
            # the losses will be separate for each batch, process
            # but the gradients accumulate across all processes.
            # therefore you dont need to only do grad.step() on the main process/sync step because it knows   
            with accelerator.accumulate(model):  # Use accelerator's context manager
                # 1. Forward pass
                # outputs = model(**batch, use_cache=False, num_items_in_batch=num_items_in_batch, new_token_start_index=original_vocab_size)
                # print("labels shape:", batch["labels"].shape if "labels" in batch else None, "loss_mask shape:", batch["loss_mask"].shape if "loss_mask" in batch else None)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],           # must be present
                    loss_mask=batch["loss_mask"],     # must be present
                    use_cache=False,
                    num_items_in_batch=num_items_in_batch,
                    new_token_start_index=original_vocab_size
                )
                # print(f"outputs: {outputs}")
                loss = outputs.loss
                loss_new_tokens = 0 # TODO need to fix this outputs.loss_new_tokens.item()
                # loss = (loss * gradient_accumulation_steps * accelerator.num_processes)

                mem_before_backward = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory before backwards pass opt_step: {update_step} accum_step:{i} - Device {accelerator.process_index}: {mem_before_backward:.2f} MB", main_process_only=False)

                accelerator.backward(loss)
                # Immediate cleanup after backward pass

                accumulated_losses.append(loss.item())  # Move loss to CPU immediately
                accumulated_losses_new_tokens.append(loss_new_tokens)
                accumulated_token_counts.append(num_items_in_batch.item())
                accumulation_batch_counter += 1
                new_token_counter.append(num_new_tokens_in_batch)

                # DO SIMPLE AVERAGE OVER ACCUMULATION STEPS
                del outputs
                del loss
                torch.cuda.empty_cache()  # Consider moving this outside the loop
                
                if accelerator.sync_gradients:
                    if num_new_tokens > 0:
                        new_embeddings_list = get_new_embeddings_grads(model, num_new_tokens)
                        new_embeddings_grad_norm = calculate_grad_norm(new_embeddings_list, is_grad=True)

                        # new_embeddings_list = get_new_embedding_params(model, num_new_tokens)
                        # new_embeddings_grad_norm = accelerator.clip_grad_norm_(new_embeddings_list, float('inf'))

                    if args.grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_norm)

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

            gathered_metrics_list = [
                accumulated_losses,
                accumulated_losses_new_tokens,
                accumulated_token_counts,
                accumulation_batch_counter,
                new_token_counter
            ]
            # All processes must participate in gather
            gathered_metrics = accelerator.gather_for_metrics(gathered_metrics_list)

            if accelerator.is_local_main_process:
                num_gathered_metrics = len(gathered_metrics)
                # Regroup the metrics by type
                gathered_losses = [item for sublist in gathered_metrics[0::num_gathered_metrics] for item in sublist]
                gathered_losses_new_tokens = [item for sublist in gathered_metrics[1::num_gathered_metrics] for item in sublist]
                gathered_tokens = [item for sublist in gathered_metrics[2::num_gathered_metrics] for item in sublist]
                gathered_batch_counter = sum(gathered_metrics[3::num_gathered_metrics])
                gathered_new_token_counter = [item for sublist in gathered_metrics[4::num_gathered_metrics] for item in sublist]
                
                # Aggregate loss
                batch_sum_weighted_loss = 0
                batch_total_tokens = 0
                for loss, tokens in zip(gathered_losses, gathered_tokens):
                    batch_sum_weighted_loss += loss * tokens
                    batch_total_tokens += tokens

                avg_loss = sum(gathered_losses) / len(gathered_losses)
                weighted_avg_loss = batch_sum_weighted_loss / batch_total_tokens
                cumulative_token_counter += batch_total_tokens
                cumulative_batch_counter += gathered_batch_counter

                # aggregate loss for new tokens
                batch_sum_weighted_loss_new_tokens = 0
                batch_total_tokens_new_tokens = 0
                for loss, tokens in zip(gathered_losses_new_tokens, gathered_new_token_counter):
                    batch_sum_weighted_loss_new_tokens += loss * tokens
                    batch_total_tokens_new_tokens += tokens

                avg_loss_new_tokens = sum(gathered_losses_new_tokens) / len(gathered_losses_new_tokens)
                weighted_avg_loss_new_tokens = batch_sum_weighted_loss_new_tokens / batch_total_tokens_new_tokens

                # Create metrics dict
                metrics_dict = {
                    "loss": avg_loss,
                    "weighted_loss": weighted_avg_loss,
                    "loss_new_tokens": avg_loss_new_tokens,
                    "weighted_loss_new_tokens": weighted_avg_loss_new_tokens,
                    "token_counter": batch_total_tokens,
                    "cum_token_counter": cumulative_token_counter,
                    "batch_counter": gathered_batch_counter,
                    "cum_batch_counter": cumulative_batch_counter,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "new_embeddings_grad_norm": new_embeddings_grad_norm,
                    "num_new_tokens": sum(gathered_new_token_counter)
                }
                metrics_dict = {f"train_{k}": v for k, v in metrics_dict.items()}
                metrics_dict["step"] = update_step
                metrics_dict["epoch"] = epoch
                loss_log = {
                    "loss": avg_loss,
                }
            
                log_loss(metrics_dict, output_dir)
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
                if save_model_type == "all" and isinstance(args.checkpointing_steps, int) and update_step > 0:
                    if update_step % args.checkpointing_steps == 0:
                        logger.info(f"Checkpointing step {update_step}")
                        # output_dir_ext = f"step_{update_step}"
                        output_dir_ext = f"intermediate_model"
                        output_dir_path = os.path.join(output_dir, output_dir_ext)
                        accelerator.save_state(output_dir_path)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            output_dir_path,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            # state_dict=state_dict,
                        )

            del gathered_metrics
    
            if update_step % args.eval_steps == 0:
                logger.debug(f"EVAL:Step {update_step}: Starting evaluation")
                move_optimizer_to_cpu(optim)
                torch.cuda.empty_cache()
                eval_metrics_dict = evaluate(model, eval_loader, accelerator, args.eval_iters, eval_config)
                move_optimizer_to_gpu(optim)
                logger.debug(f"EVAL:Step {update_step}: Eval Loss = {eval_metrics_dict['loss']:.4f}, Weighted Eval Loss = {eval_metrics_dict['weighted_loss']:.4f}")

                eval_metrics_dict = {f"eval_{k}": v for k, v in eval_metrics_dict.items()}
                eval_metrics_dict["step"] = update_step
                if accelerator.is_main_process:
                    # Use the same dict for both logging and wandb
                    log_loss(eval_metrics_dict, output_dir, filename="loss_log_eval.csv")
                    if args.wandb:
                        wandb.log(eval_metrics_dict)
                
                    logger.debug(eval_metrics_dict)
                #TODO  add early stopping
    
    logger.info(f"Training Finished")

    accelerator.clear()  # This clears the accelerator's internal state

    torch.cuda.empty_cache()
    
    # Make sure all processes are synced
    accelerator.wait_for_everyone()    

    if save_model_type is not None:
        logger.info(f"Preparing to save model to {output_dir}")
        
        if not check_disk_space(output_dir, required_space_gb=20):
            logger.info("Aborting save due to insufficient disk space")
            return
        
        # if torch.distributed.is_initialized():
        #     accelerator.wait_for_everyone()

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

        output_dir_ext = f"final_model"
        output_dir_path = os.path.join(output_dir, output_dir_ext)
        accelerator.save_state(output_dir_path)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            # state_dict=state_dict,
        )

        # Clear memory before final evaluation
        accelerator.clear()
        torch.cuda.empty_cache()
        gc.collect()

        # Move optimizer to CPU and clear GPU memory
        move_optimizer_to_cpu(optim)
        torch.cuda.empty_cache()
        
        # Run final evaluation
        eval_metrics_dict = evaluate(model, eval_loader, accelerator, args.eval_iters, eval_config)
        # move_optimizer_to_gpu(optim)
        logger.debug(f"EVAL FINAL: Eval Loss = {eval_metrics_dict['loss']:.4f}, Weighted Eval Loss = {eval_metrics_dict['weighted_loss']:.4f}")

        eval_metrics_dict = {f"eval_{k}": v for k, v in eval_metrics_dict.items()}
        eval_metrics_dict["step"] = update_step

        if accelerator.is_main_process:
            # Use the same dict for both logging and wandb
            log_loss(eval_metrics_dict, output_dir, filename="loss_log_eval.csv")
            if args.wandb:
                wandb.log(eval_metrics_dict)
            
        logger.info(f"Loss log saved to {output_dir}/loss_log.csv")
        logger.info(f"Saving Finished")

    # Clear everything before ending training
    del model
    del unwrapped_model
    del optim
    del eval_loader
    del train_loader
    accelerator.clear()
    torch.cuda.empty_cache()
    gc.collect()

    # Make sure all processes are synced before ending
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    try:
        threads = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
    except:
        threads = os.cpu_count()

    args = argparse.ArgumentParser()
    args.add_argument("--dry-run", action="store_true")
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--cpu-batch-size", type=int, default=1000)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=False, default=None)
    args.add_argument("--wandb", type=str)
    args.add_argument("--wandb-tags", type=str)
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
                     choices=["default", "random", "clone", "mean", "zeros", "merge", None],
                     default=None,
                     help="Strategy for initializing new token embeddings")
    args.add_argument("--save-model", type=str,
                     choices=["final", "all", None],
                     default="all",
                     help="Whether to save the model after training")
    args.add_argument("--finetuning-mode", type=str,
                      choices=["full", "embeddings", "new_tokens_only"],
                      default="full",
                      help="Whether to finetune the model parameters")
    args.add_argument("--task-name", type=str,
                      choices=["SFT", "translation", "mixed"],
                      default="SFT",
                      help="Whether to finetune the model parameters")

    # EVAL PARAMS
    args.add_argument("--run-lm-eval", action="store_true",
                     help="Run language model evaluation")
    args.add_argument("--eval-batch-size", type=int, default=2,
                     help="Batch size for evaluation")
    args.add_argument("--num-fewshot", type=int, default=4,
                     help="Number of fewshot examples")
    args.add_argument("--limit", type=int, default=1,
                     help="Number of samples to limit evaluation to")
    args.add_argument("--log-samples", action="store_true",
                     help="store actual samples from benchmark")
    args.add_argument("--benchmark-tasks", type=str, default="minerva_math",
                     help="Benchmark tasks to run")
    args.add_argument("--pre-tok-name", type=str, default="empty")
    
    args = args.parse_args()

    main(args)

    # TODO consider training params:
    # - grad norm
    # - lora
