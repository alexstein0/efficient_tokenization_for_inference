import argparse
import torch
import torch.nn as nn
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, ProjectConfiguration, DataLoaderConfiguration
from accelerate.state import AcceleratorState, PartialState
from tqdm import tqdm
from transformers import set_seed, DataCollatorWithPadding, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AutoModelForCausalLM, TrainingArguments, Trainer, TextStreamer, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import datasets
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import torch.distributed as dist
import csv
import shutil
import math
from typing import Any, Dict, List, Union
from collections import defaultdict

import socket
from accelerate.logging import get_logger
import logging
import json
import glob

import psutil
from efficient_tokenization.tokenize_simple import get_tokenized_data, flatten_genqa_conversations, my_tokenize, get_genqa_data, get_tokenizer
from efficient_tokenization.extend_embeddings import extend_model_embeddings, initialize_new_embeddings, get_new_embedding_params, get_new_embeddings_grads, unfreeze_model, freeze_model_except_embeddings, freeze_old_embeddings
from efficient_tokenization.data_utils import MyPaddingCollator, MyPaddingCollatorWithLossMask, create_memory_efficient_loader
from efficient_tokenization.utils import setup_logging, check_disk_space, generate_hashed_dir_name, get_cpus
from efficient_tokenization.model_utils import forward_pass, move_optimizer_to_cpu, move_optimizer_to_gpu, calculate_grad_norm, save_checkpoint, remove_old_checkpoints
from efficient_tokenization.benchmarking_utils import get_lm_eval_string

from lm_eval import evaluator, tasks, utils, models
from lm_eval.tasks import TaskManager

from dataclasses import dataclass

from liger_kernel.transformers import AutoLigerKernelForCausalLM, LigerCrossEntropyLoss

import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# def initialize_distributed():
#     if not dist.is_initialized():
#         logger.info("initializing process group")
#         dist.init_process_group(backend='nccl')  # or 'gloo' depending on your setup


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

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the checkpoints directory."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"No checkpoints directory found at {checkpoint_dir}")
        return None
    
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))
    if not checkpoint_paths:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    # Sort by modification time (most recent first)
    checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
    latest = checkpoint_paths[0]
    logger.info(f"Found latest checkpoint: {latest}")
    return latest

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

def run_evaluation_loop(model, eval_loader, accelerator, num_steps: int, loss_types: List[str] = [], materialize_logits: bool = True):
    # Set model to eval mode
    model.eval()
    
    # Wait for all processes to reach this point
    accelerator.wait_for_everyone()
    
    with torch.no_grad():
        all_gathered_losses = defaultdict(list)
        all_gathered_tokens = defaultdict(list)
        for i, batch in enumerate(eval_loader):
            if i >= num_steps:
                break
            # num_items_in_batch = (batch["labels"].ne(-100)).sum().cpu().item()

            if len(loss_types) == 1:
                # if only tracking one loss, use that loss for main loss
                main_loss_type = loss_types[0]
                loss_types = []
            else:
                main_loss_type = None

            main_loss, tracked_losses, num_items_for_main_loss, tracked_tokens_per_loss_type = forward_pass(
                model, batch, loss_with_grad=main_loss_type, losses_without_grad=loss_types, materialize_logits=materialize_logits
            )
            if main_loss_type is not None:
                tracked_losses[main_loss_type] = main_loss.item()
                tracked_tokens_per_loss_type[main_loss_type] = num_items_for_main_loss

            values_to_gather = [
                tracked_losses,
                tracked_tokens_per_loss_type
                ]
            
            num_gathered_metrics = len(values_to_gather)
            gathered_metrics = accelerator.gather_for_metrics(values_to_gather)
            gathered_losses = gathered_metrics[0::num_gathered_metrics]
            gathered_tokens = gathered_metrics[1::num_gathered_metrics]

            for loss_dict in gathered_losses:
                for key, loss_val in loss_dict.items():
                    all_gathered_losses[key].append(loss_val)
            
            for token_dict in gathered_tokens:
                for key, token_count in token_dict.items():
                    all_gathered_tokens[key].append(token_count)
                    
            del main_loss, tracked_losses, num_items_for_main_loss, tracked_tokens_per_loss_type

            # Add periodic synchronization (particularly important)
            if i % 5 == 0:  # Every 5 steps
                accelerator.wait_for_everyone()
                
            # Clear CUDA cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
        losses_dict = {}
        for loss_type in all_gathered_losses.keys():
            losses = all_gathered_losses[loss_type]
            tokens = all_gathered_tokens[loss_type]

            batch_sum_weighted_loss = 0
            batch_total_tokens = 0
            for loss, tok in zip(losses, tokens):
                batch_sum_weighted_loss += loss * tok
                batch_total_tokens += tok

            weighted_avg_loss = batch_sum_weighted_loss / batch_total_tokens if batch_total_tokens > 0 else 0
            avg_loss = sum(losses) / len(losses)

            losses_dict[f"{loss_type}_weighted_loss"] = weighted_avg_loss
            losses_dict[f"{loss_type}_loss"] = avg_loss

    # Final synchronization and cleanup
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    
    # Set model back to training mode
    model.train()
    
    return losses_dict


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
    
    eval_loop_results = run_evaluation_loop(model, eval_loader, accelerator, num_steps, eval_config.get("losses_to_track", []), eval_config.get("materialize_logits", True))

    results.update(eval_loop_results)
    model.train()
    return results


def main(args):

    global logger  # Update the logger once accelerator is initialized

    set_seed(args.seed)

    state = PartialState()
    num_processes = state.num_processes

    save_checkpoints_type = args.save_checkpoints
    logging_mode = args.logging_mode
    if args.dry_run:
        logging_mode = logging.DEBUG
        if args.wandb and args.wandb_tags:
            args.wandb_tags = ["dryrun"]
        if args.checkpointing_steps > 1:  #sometimes we are checking checkpoints every step and we dont want dry run to prevent that
            save_checkpoints_type = None

    logger = setup_logging(logging_mode)

    ###### INIT MODEL ######
    logger.info(f"Loading model from {args.model}...")
    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        # cross_entropy=True,
        # fused_linear_cross_entropy=False,
        # config=config,
        # use_cache=False,  # Disable KV cache during training
    )
    
    if hasattr(model, "enable_input_require_grads"):
        # TODO not sure what this does
        model.enable_input_require_grads()

    model.gradient_checkpointing_enable()

    ###### INIT TOKENIZER ######
    logger.info("Loading tokenizer...")
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

    original_vocab_size = model.config.vocab_size
    model.config.original_vocab_size = original_vocab_size
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}, model vocab size: {original_vocab_size}")
    num_new_tokens = len(tokenizer) - model.config.vocab_size
    args.num_new_tokens = num_new_tokens

    ###### BATCH SIZE STUFF ######
    logger.info(f"Setting batch size and gradient accumulation steps...")
    if args.total_batch_size is not None:
        total_batch_size = args.total_batch_size
        if args.batch_size is not None:
            batch_size = args.batch_size
            gradient_accumulation_steps = total_batch_size // (batch_size * num_processes)
        elif args.gradient_accumulate_every is not None:
            gradient_accumulation_steps = args.gradient_accumulate_every
            batch_size = total_batch_size // (gradient_accumulation_steps * num_processes)
        else:
            raise ValueError("Either batch_size or gradient_accumulate_every must be provided if inferring from total_batch_size")
    else:
        gradient_accumulation_steps = args.gradient_accumulate_every
        batch_size = args.batch_size
        total_batch_size = args.batch_size * args.gradient_accumulate_every * num_processes

    # make sure the rounding is correct
    total_batch_size = (
        batch_size * num_processes * gradient_accumulation_steps
    )
    num_epochs = args.num_epochs

    args.batch_size = batch_size
    args.gradient_accumulate_every = gradient_accumulation_steps
    args.total_batch_size = total_batch_size

    ###### OUTPUT FILE NAMING STUFF ######
    logger.info(f"Setting output directory with unique hash of params...")
    # ablation params are:
    model_name = args.model.split('/')[-1] # 0. model
    finetuning_params = args.finetune_params # 1. finetune_params
    embedding_init_strategy = args.embedding_init_strategy # 2. embedding_init_strategy
    dry_run = args.dry_run # 3. dry_run
    # total_batch_size = args.total_batch_size # 4. total_batch_size
    learning_rate = args.learning_rate # 5. learning_rate
    task_name = args.task_name # 6. task_name
    main_loss_type = args.main_loss # 7. main_loss
    # num_new_tokens = args.num_new_tokens # 8. num_new_tokens

    params_dict = {
        "model_name": model_name,
        "task_name": task_name,
        "finetuning_params": finetuning_params,
        "total_batch_size": total_batch_size,
        "learning_rate": learning_rate,
        "main_loss_type": main_loss_type,
        "embedding_init_strategy": embedding_init_strategy,
        "num_new_tokens": num_new_tokens
    }

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # uses hash of params to generate unique output dir
        output_dir = generate_hashed_dir_name(params_dict, dry_run=dry_run)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(params_dict, f)

    ###### INIT OPTIMIZER ######
    logger.info(f"Initializing optimizer...")
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
    logger.info(f"Initial learning rate from optimizer: {optim.param_groups[0]['lr']}")

    ###### INITIALIZE ACCELERATOR ######
    logger.info(f"Initializing accelerator...")
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    project_config = ProjectConfiguration(project_dir=output_dir, 
                                          automatic_checkpoint_naming=True
                                          )
    
    dataloader_config = DataLoaderConfiguration(
        # use_stateful_dataloader=True,
        use_seedable_sampler = True
    )

    # dataloader_config = None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        # mixed_precision="fp16",
        kwargs_handlers=[timeout],
        project_config=project_config,
        log_with="wandb" if args.wandb else None,
        dataloader_config=dataloader_config
    )
        
    ###### INIT WANDB ######
    if accelerator.is_main_process:
        if args.wandb:
            import wandb
            tags = args.wandb_tags if len(args.wandb_tags) > 0 else None
            logger.info(f"Logging to wandb with tags: {tags}")
            wandb.login()
            run = wandb.init(project=args.wandb,
                            tags=tags,
                            name=f"Training Run {output_dir.split('/')[-1]}", 
                            config=vars(args)
                            )
            run.config.update({"output_dir" : output_dir}, allow_val_change=True)
            params_dict = {f"{k}_wandb": v for k, v in params_dict.items() if k != "output_dir"}
            run.config.update(params_dict)  # add ablation params to wandb
            # TODO Resume?


    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "efficient-tokenization",
    )

    ###### INIT LOSS TRACKING ######
    logger.info(f"Initializing loss tracking...")
    other_loss_types = args.train_losses_to_track
    eval_other_loss_types = args.eval_losses_to_track
    try:
        other_loss_types.remove(main_loss_type)
    except:
        pass

    ###### EXTEND MODEL EMBEDDINGS ######
    if embedding_init_strategy is not None and num_new_tokens > 0:
        # Extend model embeddings
        logger.info(f"Extending model embeddings with strategy: {embedding_init_strategy} and adding {num_new_tokens} new tokens")
        model = extend_model_embeddings(
            model, 
            num_new_tokens, 
            init_strategy=embedding_init_strategy,
            tokenizer=tokenizer
        )
        if "new_tokens" not in eval_other_loss_types:
            eval_other_loss_types.append("new_tokens")
    else:
        # not extending embeddings
        logger.info(f"Not extending model embeddings")
        main_loss_type = "all"
        other_loss_types = []

    embedding_size_in = model.get_input_embeddings().weight.shape[0]
    embedding_size_out = model.get_output_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer.get_vocab())
    if embedding_size_in != new_vocab_size or embedding_size_out != new_vocab_size:
        raise ValueError(f"Embedding size {embedding_size_in}/{embedding_size_out} (input/output) does not match vocab size {new_vocab_size}")

    ###### FINETUNING MODES (FREEZING PARAMS) ######
    model_is_frozen = False
    if finetuning_params == "new_tokens_only":
        model = freeze_old_embeddings(model, num_new_tokens)
        model_is_frozen = True
    elif finetuning_params == "embeddings":
        model = freeze_model_except_embeddings(model)
        model_is_frozen = True
    elif finetuning_params == "full":
        pass
    else:
        raise ValueError(f"Invalid finetuning mode: {finetuning_params}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Finetuning_params: {finetuning_params}, Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

    # After model loading
    logger.info(f"Model config: max_position_embeddings={model.config.max_position_embeddings}, "
               f"vocab_size={model.config.vocab_size}")


    # postional embedding scaling
    # if hasattr(model.config, "rope_scaling"):
    #     logger.info(f"RoPE scaling config: {model.config.rope_scaling}")

    # TODO: implement task 
    ###### DATA/TASK STUFF ######
    logger.info(f"Setting up data/task stuff...")
    if task_name == "SFT":
        data_collator = MyPaddingCollator(
            tokenizer=tokenizer,
            max_length=args.max_length if hasattr(args, 'max_length') else None
        )
        if "translated" in other_loss_types:
            other_loss_types.remove("translated")

        if "translated" in eval_other_loss_types:
            eval_other_loss_types.remove("translated")
    elif task_name == "translation":
        data_collator = MyPaddingCollatorWithLossMask(
            tokenizer=tokenizer,
            max_length=args.max_length if hasattr(args, 'max_length') else None
        )
        if "translated" not in other_loss_types and main_loss_type != "translated":
            other_loss_types.append("translated")

        if "translated" not in eval_other_loss_types:
            eval_other_loss_types.append("translated")
    elif task_name == "mixed":
        # TODO: implement mixed task
        raise NotImplementedError("Mixed task is not implemented yet")
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    
    logger.info(f"Task: {task_name}, Primary loss: {main_loss_type}, Train tracked losses: {other_loss_types}, Eval tracked losses: {eval_other_loss_types}")

    ###### LOAD DATA ######
    logger.info(f"Loading data from {args.dataset}...")
    ds = load_from_disk(args.dataset)
    ds = ds.train_test_split(test_size=0.1) # Split the dataset into train (90%) and validation (10%)

    train_loader, train_sampler = create_memory_efficient_loader(
        ds["train"],
        batch_size=batch_size,
        collate_fn=data_collator,
        num_proc=args.num_proc, 
        accelerator=accelerator,
    )
    train_samples = len(ds["train"])
    train_batches = len(train_loader)
    train_loader_token_count = sum(ds["train"]["num_tokens"])

    # accelerator.register_for_checkpointing(train_sampler)

    eval_loader, eval_sampler = create_memory_efficient_loader(
        ds["test"],
        batch_size=args.eval_batch_size,  # TODO make sure this works
        collate_fn=data_collator,
        num_proc=args.num_proc,
        accelerator=accelerator
    )
    eval_samples = len(ds["test"])
    eval_batches = len(eval_loader)
    eval_loader_token_count = sum(ds["test"]["num_tokens"])

    def get_next_batch(training_iterator, train_loader, epoch, epoch_step, skip_iterator=None):
        if skip_iterator is not None:
            try:
                batch = next(skip_iterator)
                batch = {k: v.cpu() for k, v in batch.items()}
                return batch, train_loader, epoch, epoch_step, skip_iterator
            except StopIteration:
                logger.info("finished skip iter")
                epoch += 1
                epoch_step = 0
        try:
            batch = next(training_iterator)
            # Immediately move to CPU if not needed
            batch = {k: v.cpu() for k, v in batch.items()}
            return batch, training_iterator, epoch, epoch_step, None
        except StopIteration:
            # End of epoch reached, create new iterator
            epoch += 1
            logger.info(f"Starting epoch {epoch}")
            training_iterator = iter(train_loader)
            batch = next(training_iterator)
            batch = {k: v.cpu() for k, v in batch.items()}
            epoch_step = 0
            return batch, training_iterator, epoch, epoch_step, None
        
    logger.info("Loaded data into dataset")

    # if args.lora:
    #     from peft import get_peft_model, LoraConfig, TaskType
    #     target_modules = find_all_linear_names(model)
    #     my_logger(accelerator, f"LoRA target modules: {target_modules}")
    #     peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
    #                              r=16, lora_alpha=64, lora_dropout=0.05, target_modules=target_modules)
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    ###### TRACK TRAINING STEPS ######
    if args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
    else:
        max_train_steps = train_batches//(gradient_accumulation_steps * num_processes)

    # these calculations are done AFTER the accelerator is initialized
    num_batches_per_device_per_epoch = len(train_loader)
    grad_updates_per_device_per_epoch = math.ceil(num_batches_per_device_per_epoch / gradient_accumulation_steps)

    epoch_remainder_on_device = num_batches_per_device_per_epoch % gradient_accumulation_steps
    epoch_remainder_on_device = epoch_remainder_on_device if epoch_remainder_on_device != 0 else gradient_accumulation_steps

    if num_epochs > 0:
        total_gradient_updates = min(max_train_steps, grad_updates_per_device_per_epoch * num_epochs * num_processes)
        total_gradient_updates_per_device = min(math.ceil(max_train_steps // num_processes), grad_updates_per_device_per_epoch * num_epochs)
    else:
        total_gradient_updates = max_train_steps
        total_gradient_updates_per_device = math.ceil(max_train_steps // num_processes)

    ###### INIT SCHEDULER ######
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
    split_scheduler = False  # split schedule will split learning rate across multiple devices by accelerator

    logger.info("Creating objects from scratch.  Will load checkpoint if specified")
    
    ###### PREPARE ARTIFACTS TO BE USED BY ACCELERATOR ######
    logger.info(f"Preparing artifacts to be used by accelerator...")
    # prepare artifacts - accelerator handles device placement and dataloader splitting
    model, optim = accelerator.prepare(model, optim)
    train_loader = accelerator.prepare_data_loader(train_loader, device_placement=True)
    eval_loader = accelerator.prepare_data_loader(eval_loader, device_placement=True)

    if split_scheduler:
        scheduler = accelerator.prepare(scheduler)

    accelerator.register_for_checkpointing(scheduler)

    ###### INIT TRAINING STATE ######
    # samples tracking
    epoch = 0
    epoch_step = -1
    update_step = -1
    total_batched_samples = 0

    # loss tracking
    cumulative_batch_counter = 0  # number of batches processed since last accumulation
    cumulative_token_counter = 0  # all tokens processed across all processes
    cumulative_new_token_counter = 0  # all new tokens processed across all processes

    ###### LOAD CHECKPOINT ######
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoint_path = get_latest_checkpoint(output_dir)
            if checkpoint_path is None:
                logger.warning("No checkpoints found to resume from. Starting training from scratch.")
                loaded_checkpoint = False
            else:
                try:                    
                    # Load the state
                    logger.info(f"Loading accelerator state from {checkpoint_path}")
                    accelerator.load_state(checkpoint_path)
                    checkpoint_no = int(checkpoint_path.split("_")[-1]) + 1
                    accelerator.project_configuration.iteration = checkpoint_no
                    loaded_checkpoint = True
                    logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    loaded_checkpoint = False
        else:
            # Load specific checkpoint
            checkpoint_path = os.path.join(output_dir, args.resume_from_checkpoint)
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
            logger.info(f"Loading checkpoint from: {checkpoint_path}")

    logger.debug(f"Accelerator state: {accelerator.state}")

    ###### LOAD CHECKPOINT METADATA ######
    skip_iterator = None
    if loaded_checkpoint:
        # load training state
        try:
            state_dict = torch.load(os.path.join(checkpoint_path, "checkpoint_meta.pt"))
            resume_step = state_dict["update_step"]
            epoch = state_dict["epoch"]
            epoch_step = state_dict["epoch_step"]
            total_batched_samples = state_dict["total_batched_samples"]
            cumulative_batch_counter = state_dict["cumulative_batch_counter"]
            cumulative_token_counter = state_dict["cumulative_token_counter"]
            cumulative_new_token_counter = state_dict["cumulative_new_token_counter"]
            batch_skips = int((epoch_step+1)*gradient_accumulation_steps)
            if batch_skips >= len(train_loader):
                logger.warning(f"Epoch step {batch_skips} is greater than the number of batches in the train loader {len(train_loader)}")
                epoch_step = -1
            else:
                logger.info(f"Skipping {batch_skips} batches (per loader)")
                skip_loader = accelerator.skip_first_batches(train_loader, batch_skips)  # skip step in epoch
                skip_iterator = iter(skip_loader)
            update_step = resume_step
            logger.info(f"Resuming training after step {resume_step}")
        except Exception as e:
            logger.info(f"Failed to load checkpoint metadata: {checkpoint_path}")
            logger.info(f"Error: {e}")

    training_iterator = iter(train_loader)

    logger.info(f"Setup Eval config...")
    materialize_logits = not args.do_not_materialize_logits
    if args.benchmark_tasks is not None:
        if isinstance(args.benchmark_tasks, str):
            task_list = args.benchmark_tasks.split(",")
        else:
            task_list = args.benchmark_tasks
    else:
        logger.info("No benchmark tasks specified, skipping evaluation")
        task_list = None

    eval_config = {
        "run_lm_eval": args.run_lm_eval,
        "batch_size": args.eval_batch_size,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "log_samples": args.log_samples,
        "benchmark_tasks": task_list,
        "losses_to_track": eval_other_loss_types,
        "materialize_logits": materialize_logits,
    }

    eval_config["tokenizer"] = tokenizer
    if base_tokenizer is not None:
        eval_config["base_tokenizer"] = base_tokenizer
    if vocab_file_path is not None:
        eval_config["tokenizer_path"] = vocab_file_path
    if pre_tok_name is not None:
        eval_config["pre_tok_name"] = pre_tok_name

    logger.info(f"--------------------------------")
    logger.info(f"Training setup:")
    logger.info(f"max train steps {max_train_steps}") # ({math.ceil(max_train_steps / num_processes)} per device)")
    logger.info(f"total gradient updates {total_gradient_updates} (per epoch {grad_updates_per_device_per_epoch * num_processes})") # ({total_gradient_updates_per_device} per device)")
    logger.info(f"gradient steps {gradient_accumulation_steps}")
    logger.info(f"per device batch size {batch_size}, eval batch size {args.eval_batch_size}")
    logger.info(f"num gpus: {num_processes}")
    logger.info(f"effective batch size {total_batch_size}")
    logger.info(f"learning rate {args.learning_rate}, warmup steps {args.warmup_steps}, schedule_max_steps {max_train_steps}, split_scheduler {split_scheduler}")
    logger.info(f"checkpointing steps {args.checkpointing_steps}")
    logger.info(f"Training until {max_train_steps} steps")

    logger.info(f"train samples: {train_samples}, batches: {train_batches}, and tokens: {train_loader_token_count}")
    logger.info(f"train samples per device: {len(train_loader) * batch_size}, batches per device: {len(train_loader)}, and tokens (approx): {train_loader_token_count / num_processes}")
    logger.info(f"eval samples: {eval_samples}, batches: {eval_batches}, and tokens: {eval_loader_token_count}")
    logger.info(f"eval samples per device: {len(eval_loader) * args.eval_batch_size}, batches per device: {len(eval_loader)}, and tokens (approx): {eval_loader_token_count / num_processes}")
    logger.info(f"Task: {task_name}, extending params: {embedding_init_strategy}")
    
    logger.info(f"accelerator distributed type {accelerator.distributed_type}, num_processes {num_processes} on {torch.cuda.get_device_name()}")
    logger.info(f"CPUs: {args.num_proc}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
    logger.info(f"--------------------------------")

    can_train = True
    if args.save_only:
        can_train = False

    ###### TRAINING LOOP ######
    logger.info(f"Starting training loop...")
    logger.info("")
    use_progress_bar = True
    progress_bar = tqdm(
        range(total_gradient_updates),
        disable=(not use_progress_bar) or (not accelerator.is_local_main_process),  # turning off progress bar
        initial=update_step + 1,
    )

    largest_batch_per_device = 0
    while update_step < total_gradient_updates - 1:
        epoch_step += 1
        update_step += 1
        logger.debug(f"Starting step {update_step} of {total_gradient_updates}")
        
        if not can_train:
            break

        mem_before = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"Memory before loading batches - Device {accelerator.process_index}: {mem_before:.2f} MB", main_process_only=False)

        model.train()
        # TODO add time of each iteration, and memory usage
        if model_is_frozen and args.unfreeze_params_steps > 0 and update_step >= args.unfreeze_params_steps:
            logger.info(f"Unfreezing model parameters at step {update_step}")
            model = unfreeze_model(model)
            model_is_frozen = False
            
        loss_log = None
        grad_norm = None
        new_embeddings_grad_norm = None
        
        num_batches_in_step = gradient_accumulation_steps if epoch_step != (grad_updates_per_device_per_epoch - 1) else epoch_remainder_on_device
        logger.debug(f"gradient accumulation steps: {gradient_accumulation_steps} total gradient updates: {total_gradient_updates}, remainder: {epoch_remainder_on_device}, num_batches_in_step {num_batches_in_step}")

        # get local num items in batch
        device_memory_usage = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"Step {update_step} - Device {accelerator.process_index} - device memory usage {device_memory_usage} MB, largest_batch_per_device {largest_batch_per_device}", main_process_only=False)

        accumulation_batch_counter = 0
        new_token_counter = 0
        total_token_counter = 0
        accumulated_losses_per_loss_type = defaultdict(list)
        accumulated_tokens_per_loss_type = defaultdict(list)
        
        for i in range(num_batches_in_step):
            batch, training_iterator, epoch, epoch_step, skip_iterator = get_next_batch(training_iterator, train_loader, epoch, epoch_step, skip_iterator)
            total_batched_samples += 1

            # Move batch to device just before use
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            mem_before_forward = torch.cuda.memory_allocated() / 1024**2
            largest_batch_per_device = max(largest_batch_per_device, batch['input_ids'].numel())
            num_items_in_batch = (batch["labels"].ne(-100)).sum(1).cpu().tolist()


            logger.debug(f"Memory before forward pass opt_step: {update_step} accum_step:{i} - Device {accelerator.process_index}: {mem_before_forward:.2f} MB, "
                         f"largest_batch_per_device {largest_batch_per_device} "
                         f"ids: {batch['input_ids'].numel()}, tokens with losses: {num_items_in_batch}", main_process_only=False
                         )

            # IMPORTANT:
            # the losses will be separate for each batch, process
            # but the gradients accumulate across all processes.
            # therefore you dont need to only do grad.step() on the main process/sync step because it knows   
            with accelerator.accumulate(model):  # Use accelerator's context manager
                main_loss, tracked_losses, num_items_for_loss, tracked_tokens_per_loss_type = forward_pass(
                    model, batch, loss_with_grad=main_loss_type, losses_without_grad=other_loss_types, materialize_logits=materialize_logits
                )
                model.train()

                mem_before_backward = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory before backwards pass opt_step: {update_step} accum_step:{i} - Device {accelerator.process_index}: {mem_before_backward:.2f} MB", main_process_only=False)
                accelerator.backward(main_loss)

                # just copy this into the dict so we have all the info there
                tracked_losses[main_loss_type] = main_loss.item()
                tracked_tokens_per_loss_type[main_loss_type] = num_items_for_loss

                num_new_tokens_in_batch = tracked_tokens_per_loss_type.get("new_tokens", 0)
                num_items_in_batch = tracked_tokens_per_loss_type.get("all", 0)

                accumulation_batch_counter += 1
                new_token_counter += num_new_tokens_in_batch
                total_token_counter += num_items_in_batch
                for loss_type in tracked_losses:
                    accumulated_losses_per_loss_type[loss_type].append(tracked_losses[loss_type])
                    accumulated_tokens_per_loss_type[loss_type].append(tracked_tokens_per_loss_type[loss_type])

                del main_loss, tracked_losses, num_items_for_loss, tracked_tokens_per_loss_type
                # torch.cuda.empty_cache()  # Consider moving this outside the loop
                
                if accelerator.sync_gradients:
                    if num_new_tokens > 0:
                        new_embeddings_list = get_new_embeddings_grads(model, num_new_tokens)
                        new_embeddings_grad_norm = calculate_grad_norm(new_embeddings_list, is_grad=True)

                    if args.grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_norm)

                optim.step()
                # TODO look at single neuron and gradient
                if accelerator.sync_gradients:
                    logger.debug(f"Device {accelerator.process_index} - Step {update_step}, {accumulation_batch_counter}: {scheduler.get_last_lr()}")
                    scheduler.step()
                    
                optim.zero_grad()
    
        torch.cuda.empty_cache()
                
        if accelerator.sync_gradients:
            # TODO its possible everything below here does not need to be inside a sync point since it leaves the loop
            logger.debug(f"SYNC POINT - Steps {update_step}, "
                        f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}, "
                        f"Gradient state steps: {accelerator.gradient_state.num_steps}, "
                        f"Our counter: {accumulation_batch_counter}")

            gathered_metrics_list = [
                accumulation_batch_counter,
                new_token_counter,
                total_token_counter,
                accumulated_losses_per_loss_type,
                accumulated_tokens_per_loss_type,
            ]

            # All processes must participate in gather
            gathered_metrics = accelerator.gather_for_metrics(gathered_metrics_list)

            if accelerator.is_main_process:
                num_gathered_metrics = len(gathered_metrics_list)

                # Regroup the metrics by type
                gathered_batch_counter = sum(gathered_metrics[0::num_gathered_metrics])
                gathered_new_token_counter = sum(gathered_metrics[1::num_gathered_metrics])
                gathered_total_token_counter = sum(gathered_metrics[2::num_gathered_metrics])

                all_gathered_losses = defaultdict(list)
                all_gathered_tokens = defaultdict(list)

                gathered_losses = gathered_metrics[3::num_gathered_metrics]
                for loss_dict in gathered_losses:
                    for loss_type, loss_val in loss_dict.items():
                        all_gathered_losses[loss_type].extend(loss_val)

                gathered_tokens = gathered_metrics[4::num_gathered_metrics]
                for token_dict in gathered_tokens:
                    for loss_type, token_count in token_dict.items():
                        all_gathered_tokens[loss_type].extend(token_count)
                
                losses_dict = {}
                for loss_type in all_gathered_losses.keys():
                    losses = all_gathered_losses[loss_type]
                    tokens = all_gathered_tokens[loss_type]

                    batch_sum_weighted_loss = 0
                    batch_total_tokens = 0
                    for loss, tok in zip(losses, tokens):
                        batch_sum_weighted_loss += loss * tok
                        batch_total_tokens += tok

                    weighted_avg_loss = batch_sum_weighted_loss / batch_total_tokens
                    avg_loss = sum(losses) / len(losses)

                    losses_dict[f"{loss_type}_weighted_loss"] = weighted_avg_loss
                    losses_dict[f"{loss_type}_loss"] = avg_loss


                cumulative_token_counter += gathered_total_token_counter
                cumulative_batch_counter += gathered_batch_counter
                cumulative_new_token_counter += gathered_new_token_counter


                # Create metrics dict
                metrics_dict = {
                    # "loss": avg_loss,
                    # "weighted_loss": weighted_avg_loss,
                    **losses_dict,
                    "token_counter": gathered_total_token_counter,
                    "cum_token_counter": cumulative_token_counter,
                    "new_token_counter": gathered_new_token_counter,
                    "cum_new_token_counter": cumulative_new_token_counter,
                    "batch_counter": gathered_batch_counter,
                    "cum_batch_counter": cumulative_batch_counter,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "new_embeddings_grad_norm": new_embeddings_grad_norm,
                }
                metrics_dict = {f"train_{k}": v for k, v in metrics_dict.items()}
                metrics_dict["step"] = update_step
                metrics_dict["epoch"] = epoch
                loss_log = {k: v for k, v in losses_dict.items() if not k.endswith("_weighted_loss")}
            
                log_loss(metrics_dict, output_dir)
                if args.wandb:
                    wandb.log(metrics_dict)
            
                # only update progress bar on main process also only do checkpointing on main process
                if loss_log is not None:
                    if use_progress_bar:
                        progress_bar.set_postfix(loss_log)  # This will now show both raw loss and moving average
                    else:
                        logger.info(metrics_dict)
                
                del gathered_metrics
                del metrics_dict
                del all_gathered_losses
                del all_gathered_tokens
                del losses_dict
                del loss_log
                gc.collect()

            # checkpointing (outsize main process)
            if save_checkpoints_type is not None and isinstance(args.checkpointing_steps, int) and update_step > 0:
                if update_step % args.checkpointing_steps == 0:
                    logger.info(f"Checkpointing at update step {update_step}")
                    
                    state_dict = {
                        "update_step": update_step,
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "update_step": update_step,
                        "total_batched_samples": total_batched_samples,
                        "cumulative_batch_counter": cumulative_batch_counter,
                        "cumulative_token_counter": cumulative_token_counter,
                        "cumulative_new_token_counter": cumulative_new_token_counter,
                        "current_batch_size": batch_size, # used to skip first batches
                    }
                    save_checkpoint(accelerator, output_dir, state_dict, logger, delete_old_checkpoints=not save_checkpoints_type == "all")  # delete old checkpoints if not saving all
    
            if update_step % args.eval_steps == 0:
                logger.debug(f"EVAL:Step {update_step}: Starting evaluation")
                move_optimizer_to_cpu(optim)
                torch.cuda.empty_cache()
                eval_metrics_dict = evaluate(model, eval_loader, accelerator, args.eval_iters, eval_config)
                move_optimizer_to_gpu(optim)
                log_dict = {k: v for k, v in eval_metrics_dict.items() if not k.endswith("_weighted_loss")}
                logger.debug(f"EVAL:Step {update_step}: Eval Loss = {log_dict[f'{main_loss_type}_loss']:.4f}")

                eval_metrics_dict = {f"eval_{k}": v for k, v in eval_metrics_dict.items()}
                eval_metrics_dict["step"] = update_step
                if accelerator.is_main_process:
                    # Use the same dict for both logging and wandb
                    log_loss(eval_metrics_dict, output_dir, filename="loss_log_eval.csv")
                    if args.wandb:
                        wandb.log(eval_metrics_dict)
                
                    logger.debug(eval_metrics_dict)
                #TODO  add early stopping
            if accelerator.is_main_process:
                progress_bar.update(1)
        accumulated_losses_per_loss_type.clear()
        accumulated_tokens_per_loss_type.clear()
    
    logger.info(f"Training Finished")
    
    # Make sure all processes are synced
    accelerator.wait_for_everyone()    

    if save_checkpoints_type is not None:
        logger.info(f"Preparing to save model to {output_dir}")
        
        if not check_disk_space(output_dir, required_space_gb=20):
            logger.info("Aborting save due to insufficient disk space")
            return

        output_dir_ext = f"final_model"
        output_dir_path = os.path.join(output_dir, output_dir_ext)

        if save_checkpoints_type == "model_only":
            if accelerator.is_main_process:
                remove_old_checkpoints(output_dir, logger)
        else:
            state_dict = {
                "update_step": update_step,
                "epoch": epoch,
                "epoch_step": epoch_step,
                "update_step": update_step,
                "total_batched_samples": total_batched_samples,
                "cumulative_batch_counter": cumulative_batch_counter,
                "cumulative_token_counter": cumulative_token_counter,
                "cumulative_new_token_counter": cumulative_new_token_counter,
                "current_batch_size": batch_size,  # used to skip first batches
            }
            save_checkpoint(accelerator, output_dir, state_dict, logger, delete_old_checkpoints=not save_checkpoints_type == "all")

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            # state_dict=state_dict,
        )
        lm_eval_string = get_lm_eval_string(output_dir_path, 
                                            tokenizer.name_or_path, 
                                            tasks=eval_config["benchmark_tasks"],
                                            # num_fewshot=eval_config["num_fewshot"],
                                            limit=eval_config["limit"],
                                            log_samples=eval_config["log_samples"],
                                            # cache_requests=eval_config["cache_requests"],
                                            # show_config=eval_config["show_config"],
                                            )
        
        with open(os.path.join(output_dir, "lm_eval.sh"), "w") as f:
            f.write(lm_eval_string)

        # Clear memory before final evaluation
        accelerator.clear()
        torch.cuda.empty_cache()
        gc.collect()
        del unwrapped_model

        # Move optimizer to CPU and clear GPU memory
        move_optimizer_to_cpu(optim)
        torch.cuda.empty_cache()
        
        # Run final evaluation
        eval_metrics_dict = evaluate(model, eval_loader, accelerator, args.eval_iters, eval_config)
        # move_optimizer_to_gpu(optim)
        log_dict = {k: v for k, v in eval_metrics_dict.items() if not k.endswith("_weighted_loss")}
        logger.debug(f"EVAL:Step {update_step}: Eval Loss = {log_dict[f'{main_loss_type}_loss']:.4f}")

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
    del optim
    del eval_loader
    del train_loader
    gc.collect()

    # Make sure all processes are synced before ending
    accelerator.wait_for_everyone()
    accelerator.end_training()

def parse_args():
    threads = get_cpus()

    args = argparse.ArgumentParser()

    # implementation params
    args.add_argument("--dry-run", action="store_true")
    args.add_argument("--logging-mode", type=str, choices=["INFO", "DEBUG"], default="INFO")
    args.add_argument("--cpu-batch-size", type=int, default=1000)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--resume-from-checkpoint", type=str, default="latest",
                     help="Path to checkpoint directory or 'latest' to let accelerate handle latest")
    args.add_argument("--wandb", type=str)
    args.add_argument("--wandb-tags", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--output-dir", type=str, required=False, default=None)
    args.add_argument("--num-proc", type=int, default=threads)
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--save-dataset-path", type=str)
    args.add_argument("--save-checkpoints", type=str,
                    choices=["final", "all", None, "model_only"],
                    default="model_only",
                    help="Whether to save the model after training")

    # finetuning params
    args.add_argument("--dataset", type=str, default=None)
    args.add_argument("--total-batch-size", type=int, default=None)
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--max-train-steps", type=int, default=-1)
    args.add_argument("--num-epochs", type=int, default=-1)
    args.add_argument("--grad-norm", type=float, default=1.0, help="Max norm for gradient clipping. Set to None to disable.")
    args.add_argument("--lora", action="store_true")  # TODO
    args.add_argument("--deepspeed", action="store_true")  # TODO
    args.add_argument("--task-name", type=str,
                      choices=["SFT", "translation", "mixed"],
                      default="SFT",
                      help="Whether to finetune the model parameters")
    
    # positional embeddings
    args.add_argument("--scaling-factor", type=float, default=16.0) # TODO
    args.add_argument("--scaling-type", type=str, default="yarn") # TODO
    args.add_argument("--rope-theta", type=float, default=10000.0) # TODO
    args.add_argument("--original-max-position-embeddings", type=int)
    args.add_argument("--max-position-embeddings", type=int)
    # LR 
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--warmup-steps", type=int, default=10)
    args.add_argument("--lr-schedule", type=str, choices=["linear", "constant"], default="linear")
    
    # Model params
    args.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
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
    args.add_argument("--finetune-params", type=str,
                      choices=["full", "embeddings", "new_tokens_only"],
                      default="full",
                      help="Whether to finetune the model parameters")
    args.add_argument("--unfreeze-params-steps", type=int,
                    default=-1,
                    help="Steps to switch to finetuning params")
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


    # EVAL PARAMS
    args.add_argument("--eval-steps", type=int, default=1000,
                     help="Number of steps between evaluations")
    args.add_argument("--eval-iters", type=int, default=100,
                     help="Number of iterations to run for evaluation")
    args.add_argument("--run-lm-eval", action="store_true",
                     help="Run language model evaluation")
    args.add_argument("--eval-batch-size", type=int, default=2,
                     help="Batch size for evaluation")
    args.add_argument("--num-fewshot", type=int, default=4,
                     help="Number of fewshot examples")
    args.add_argument("--limit", type=int, default=100,
                     help="Number of samples to limit evaluation to")
    args.add_argument("--log-samples", action="store_true",
                     help="store actual samples from benchmark")
    args.add_argument("--benchmark-tasks", type=str, default="minerva_math",
                     help="Benchmark tasks to run")
    args.add_argument("--do-not-materialize-logits", action="store_true",
                     help="Whether to not materialize logits for additional time savings")
    
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
            raise NotImplementedError("Mixed task name not implemented")
        else:
            raise ValueError(f"Invalid task name: {args.task_name}")

if __name__ == "__main__":
    
    main(parse_args())

    # TODO consider training params:
    # - lora
    # - checkpointing
