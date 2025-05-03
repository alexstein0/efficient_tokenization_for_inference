import torch
import os
import sys
from datasets import load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig, ShardingStrategy, BackwardPrefetch, StateDictType
from accelerate.utils import InitProcessGroupKwargs, set_seed, ProjectConfiguration, DataLoaderConfiguration, release_memory
from accelerate.state import PartialState
from tqdm import tqdm
from transformers import set_seed, AutoTokenizer, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import datasets
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
import torch.distributed as dist
import csv
import shutil
import math
from typing import Dict, List
from collections import defaultdict
import time

import socket
import logging
import json
import glob

import psutil
from efficient_tokenization.tokenize_simple import get_tokenizer
from efficient_tokenization.extend_embeddings import extend_model_embeddings, get_new_embeddings_grads, get_old_embeddings_grads, unfreeze_model, freeze_model_except_embeddings, freeze_old_embeddings, unfreeze_embeddings, unfreeze_first_last_layer, get_new_embedding_params, get_old_embedding_params
from efficient_tokenization.data_utils import MyPaddingCollator, MyPaddingCollatorWithLossMask, MyPaddingCollatorGeneral, create_memory_efficient_loader, load_mixed_dataset
from efficient_tokenization.utils import setup_logging, check_disk_space, generate_hashed_dir_name, get_cpus, parse_args, get_latest_checkpoint
from efficient_tokenization.model_utils import forward_pass, move_optimizer_to_cpu, move_optimizer_to_gpu, save_checkpoint, remove_old_checkpoints, calc_batch_size_stuff, setup_lora, calculate_norm, save_training_state_dict
from efficient_tokenization.benchmarking_utils import get_lm_eval_string, convert_results_into_new_token_metrics
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


from lm_eval import evaluator, tasks, utils, models
from lm_eval.tasks import TaskManager

from dataclasses import dataclass

from liger_kernel.transformers import AutoLigerKernelForCausalLM

import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def get_next_batch(ds_iterator, ds_loader, epoch, epoch_step, skip_iterator=None):
    if skip_iterator is not None:
        try:
            batch = next(skip_iterator)
            batch = {k: v.cpu() if k != "task_type" else v for k, v in batch.items()}
            return batch, ds_iterator, epoch, epoch_step, skip_iterator
        except StopIteration:
            logger.info(f"finished skip iter")
            logger.info(f"Starting epoch {epoch}")
            epoch += 1
            epoch_step = 0
    try:
        batch = next(ds_iterator)
        # Immediately move to CPU if not needed
        batch = {k: v.cpu() if k != "task_type" else v for k, v in batch.items()}
        return batch, ds_iterator, epoch, epoch_step, None
    except StopIteration:
        # End of epoch reached, create new iterator
        epoch += 1
        logger.info(f"Starting epoch {epoch}")
        ds_iterator = iter(ds_loader)
        batch = next(ds_iterator)
        batch = {k: v.cpu() if k != "task_type" else v for k, v in batch.items()}
        epoch_step = 0
        return batch, ds_iterator, epoch, epoch_step, None

def run_benchmark_loop(accelerator, model, config):
    """
    Run lm-evaluation-harness math benchmark evaluation
    """
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    model_args_dict = {
        "pretrained": unwrapped_model,  # This will be your model object directly
        # "tokenizer": config["tokenizer_path"],  # This will be your tokenizer object directly
        "tokenizer": config["tokenizer"],  # This will be your tokenizer object directly
        # "old_tokenizer": config["base_tokenizer"],  # This will be your tokenizer object directly
        # "pre_tok_name": config["pre_tok_name"],
        # "parallelize": True,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
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

    task_list = config["benchmark_tasks"]
    task_manager = TaskManager()
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

    LM_model._rank = accelerator.process_index
    LM_model._world_size = accelerator.num_processes
    limit = config["limit"] if config["limit"] > 0 else None
    log_samples = True # must be true for compression stats
    apply_chat_template = True

    results = evaluator.simple_evaluate_from_loop(
        model=LM_model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=config["num_fewshot"],
        batch_size=config["batch_size"],
        # max_batch_size=max_batch_size,
        # device=device,
        # use_cache=config.get("use_cache", None),
        limit=limit,
        # check_integrity=check_integrity,
        # write_out=write_out,
        log_samples=log_samples,
        # evaluation_tracker=evaluation_tracker,
        # system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
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

    accelerator.wait_for_everyone()
    # output_path = f"./eval_results{"" if config["experiment"] == None or len(config["experiment"]) == 0 else f"/{config['experiment']}"}"
    # TODO SAVE predictions

    outputs = convert_results_into_new_token_metrics(results, task_names, config["tokenizer"], config["base_tokenizer"])

    metrics_dict = outputs[task_names[0]]["metrics"]
    model_token_counts = outputs[task_names[0]]["model_token_counts"]
    theoretical_token_counts = outputs[task_names[0]]["theoretical_token_counts"]
    old_theoretical_token_counts = outputs[task_names[0]]["old_theoretical_token_counts"]
    new_token_counts = outputs[task_names[0]]["new_token_counts"]

    gathered_list = [metrics_dict, model_token_counts, theoretical_token_counts, old_theoretical_token_counts, new_token_counts]
    gathered_metrics = accelerator.gather_for_metrics(gathered_list)
    
    all_metrics_dict = defaultdict(list)
    output_dict = {}
    if accelerator.is_main_process:
        num_gathered_metrics = len(gathered_list)

        # Regroup the metrics by type
        metrics_dict = gathered_metrics[0::num_gathered_metrics]
        for process_dict in metrics_dict:
            for key, value in process_dict.items():
                all_metrics_dict[key].extend([value])
        for key, value in all_metrics_dict.items():
            output_dict[key] = sum(value) / len(value)
        model_token_list = [item for sublist in gathered_metrics[1::num_gathered_metrics] for item in sublist]
        theoretical_token_list = [item for sublist in gathered_metrics[2::num_gathered_metrics] for item in sublist]
        old_theoretical_token_list = [item for sublist in gathered_metrics[3::num_gathered_metrics] for item in sublist]
        new_token_list = [item for sublist in gathered_metrics[4::num_gathered_metrics] for item in sublist]
        
        # the bigger one goes second
        compression_ratio = [((y - x) / x) if x != 0 else 0 for x, y in zip(model_token_list, old_theoretical_token_list)]
        learning_ratio = [((y - x) / x) if x != 0 else 0 for x, y in zip(theoretical_token_list, model_token_list)]
        theoretical_compression_ratio = [((y - x) / x) if x != 0 else 0 for x, y in zip(theoretical_token_list, old_theoretical_token_list)]
        pct_new_tokens = [(x / y) if y != 0 else 0 for x, y in zip(new_token_list, model_token_list)]

        output_dict["compression_ratio"] = sum(compression_ratio) / len(compression_ratio)
        output_dict["learning_ratio"] = sum(learning_ratio) / len(learning_ratio)
        output_dict["theoretical_compression_ratio"] = sum(theoretical_compression_ratio) / len(theoretical_compression_ratio)
        output_dict["pct_new_tokens"] = sum(pct_new_tokens) / len(pct_new_tokens)
    return output_dict


def run_evaluation_loop(model, eval_iterator, eval_loader, accelerator, num_iters: int, loss_types: List[str] = [], materialize_logits: bool = True):
    # Set model to eval mode
    model.eval()
    
    # Wait for all processes to reach this point
    accelerator.wait_for_everyone()
    logger.debug(f"running eval - Device: {accelerator.process_index}", main_process_only=False)

    if len(loss_types) == 1:
        # if only tracking one loss, use that loss for main loss
        main_loss_type = loss_types[0]
        loss_types_to_track = []
    else:
        main_loss_type = None
        loss_types_to_track = loss_types
    
    # try:
    if True:
        with torch.no_grad():
            all_gathered_losses = defaultdict(list)
            all_gathered_tokens = defaultdict(list)
            
            device_losses = defaultdict(list)
            device_tokens = defaultdict(list)
            # Get total steps to process
            logger.debug(f"Device: {accelerator.process_index}, total steps: {num_iters}, steps in eval loader: {len(eval_loader)}", main_process_only=False)
            i = 0
            while i < num_iters:
                batch, eval_iterator, _, _, _ = get_next_batch(eval_iterator, eval_loader, epoch = -1, epoch_step = i, skip_iterator=None)
                mem_before = torch.cuda.memory_allocated() / 1024**2
                batch = {k: v.to(accelerator.device) if k != "task_type" else v for k, v in batch.items()}
                sum_num_items = sum(v.numel() for k, v in batch.items() if k != "task_type")
                logger.debug(f"Memory before batches {i} - Device {accelerator.process_index}: {mem_before:.2f} MB.  Batch has {sum_num_items} items", main_process_only=False)
                
                main_loss, tracked_losses, num_items_for_main_loss, tracked_tokens_per_loss_type = forward_pass(
                    model, batch, loss_with_grad=main_loss_type, losses_without_grad=loss_types_to_track, 
                    materialize_logits=materialize_logits
                )
                
                if main_loss_type is not None:
                    tracked_losses[main_loss_type] = main_loss.item()
                    tracked_tokens_per_loss_type[main_loss_type] = num_items_for_main_loss

                for loss_type in tracked_losses:
                    device_losses[loss_type].append(tracked_losses[loss_type])
                    device_tokens[loss_type].append(tracked_tokens_per_loss_type[loss_type])

                # Make sure to delete tensors to free memory
                del main_loss, tracked_losses, tracked_tokens_per_loss_type, num_items_for_main_loss, batch
                torch.cuda.empty_cache()  # Add this to clear GPU memory
                i += 1

            # Create simple, serializable values to gather
            values_to_gather = [{
                "losses": {k: [float(v) for v in vals] for k, vals in device_losses.items()},
                "tokens": {k: [int(v) for v in vals] for k, vals in device_tokens.items()}
            }]

            # outside loop
            # First synchronize all processes
            accelerator.wait_for_everyone()

            # Gather metrics across processes
            gathered_metrics = accelerator.gather_for_metrics(values_to_gather)
            
            # Process gathered metrics on main process only
            if accelerator.is_main_process:
                for proc_data in gathered_metrics:
                    for loss_type, loss_vals in proc_data["losses"].items():
                        all_gathered_losses[loss_type].extend(loss_vals)
                    for token_type, token_vals in proc_data["tokens"].items():
                        all_gathered_tokens[token_type].extend(token_vals)
            
            # Calculate final metrics
            losses_dict = {}
            for loss_type in all_gathered_losses.keys():
                try:
                    losses = all_gathered_losses[loss_type]
                    tokens = all_gathered_tokens[loss_type]

                    batch_sum_weighted_loss = 0
                    batch_total_tokens = 0
                    for loss, tok in zip(losses, tokens):
                        batch_sum_weighted_loss += loss * tok
                        batch_total_tokens += tok

                    weighted_avg_loss = batch_sum_weighted_loss / batch_total_tokens if batch_total_tokens > 0 else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0

                    losses_dict[f"{loss_type}_weighted_loss"] = weighted_avg_loss
                    losses_dict[f"{loss_type}_loss"] = avg_loss
                except Exception as e:
                    logger.info(f"Device {accelerator.process_index} - Error calculating metrics for {loss_type}: {str(e)}", main_process_only=False)
    
    # except Exception as e:
    #     logger.info(f"Error in evaluation loop: {str(e)}", main_process_only=False)
    #     import traceback
    #     logger.info(traceback.format_exc(), main_process_only=False)
    #     # Return empty dict if evaluation fails
    #     return {}
    
    # Final synchronization and cleanup
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    
    # Explicitly delete large variables
    del all_gathered_losses, all_gathered_tokens, device_losses, device_tokens
    
    # Run garbage collection
    gc.collect()
    
    # Set model back to training mode
    return losses_dict



def evaluate(model, eval_iterator, eval_loader, accelerator, eval_config: Dict = None, will_benchmark: bool = True, will_eval: bool = True):
    model.eval()
    results = {}
    if will_benchmark:
        benchmark_metrics_dict = run_benchmark_loop(accelerator, model, eval_config)
        results.update(benchmark_metrics_dict)
        accelerator.wait_for_everyone()
    if will_eval:
        eval_metrics_dict = run_evaluation_loop(
            model = model, 
            eval_iterator = eval_iterator, 
            eval_loader = eval_loader,
            accelerator = accelerator, 
            num_iters = eval_config.get("eval_iters", 100), 
            loss_types = eval_config.get("losses_to_track", []), 
            materialize_logits = eval_config.get("materialize_logits", True)
        )
        results.update(eval_metrics_dict)
    model.train()
    return results


def main(args):

    global logger  # Update the logger once accelerator is initialized
    can_train = not args.save_only

    set_seed(args.seed)
    state = PartialState()
    num_processes = state.num_processes    

    save_checkpoints_type = args.save_checkpoints
    logging_mode = args.logging_mode
    dry_run = args.dry_run

    if dry_run:
        logging_mode = logging.DEBUG
        if args.wandb and args.wandb_tags:
            args.wandb_tags = ["dryrun"]
        if args.checkpointing_steps > 1:  #sometimes we are checking checkpoints every step and we dont want dry run to prevent that
            save_checkpoints_type = None

    logger = setup_logging(logging_mode)
    logger.info("Running with following command:")
    command = " ".join(sys.argv)
    command = f"accelerate launch --num_processes {num_processes} {command}"
    logger.info(command)

    # initialize objects and do checks to make sure they are correct
    num_epochs = args.num_epochs

    ###### BATCH SIZE STUFF ######
    logger.info(f"Setting batch size and gradient accumulation steps...")
    total_batch_size, batch_size, gradient_accumulation_steps = calc_batch_size_stuff(total_batch_size = args.total_batch_size, 
                                                                                      batch_size = args.batch_size, 
                                                                                      num_processes = num_processes, 
                                                                                      gradient_accumulate_every = args.gradient_accumulate_every
                                                                                      )

    ###### OUTPUT FILE NAMING STUFF ######
    logger.info(f"Setting output directory with unique hash of params...")
    # ablation params are:
    model_name = args.model.split('/')[-1] # 0. model
    tokenizer_path = args.tokenizer_path # 1. tokenizer_path
    finetune_params = args.finetune_params # 1. finetune_params
    embedding_init_strategy = args.embedding_init_strategy # 2. embedding_init_strategy
    # dry_run = args.dry_run # 3. dry_run
    # total_batch_size = args.total_batch_size # 4. total_batch_size
    learning_rate = args.learning_rate # 5. learning_rate
    task_name = args.task_name # 6. task_name
    main_loss_type = args.main_loss # 7. main_loss
    num_new_tokens = args.num_new_tokens # 8. num_new_tokens
    # prefreeze params
    if args.unfreeze_params_steps is None or args.unfreeze_params_steps < 0 or args.finetune_params_prefreeze == args.finetune_params:
        logger.info(f"Unfreezing params after unfreeze step is not set or is the same as finetune params after unfreeze, so we will not unfreeze params")
        finetune_params_prefreeze = None
        reset_optimizer = False
        unfreeze_params_steps = -1
        warmup_steps_prefreeze = -1
        lr_schedule_prefreeze = None
        will_unfreeze_params = False
    else:
        finetune_params_prefreeze = args.finetune_params_prefreeze
        reset_optimizer = args.reset_optimizer
        unfreeze_params_steps = args.unfreeze_params_steps
        warmup_steps_prefreeze = args.warmup_steps_prefreeze if args.warmup_steps_prefreeze > -1 else unfreeze_params_steps // 10
        lr_schedule_prefreeze = args.lr_schedule_prefreeze  # this is not part of logging
        will_unfreeze_params = True

    dataset_str = args.dataset
    seed = args.seed
    warmup_steps = args.warmup_steps
    lr_schedule = args.lr_schedule

    params_dict = {
        "model_name": model_name,
        "task_name": task_name,
        "finetuning_params": finetune_params,
        "total_batch_size": total_batch_size,
        "learning_rate": learning_rate,
        "main_loss_type": main_loss_type,
        "embedding_init_strategy": embedding_init_strategy,
        "num_new_tokens": num_new_tokens,
        "unfreeze_params_steps": unfreeze_params_steps,
        "finetune_params_prefreeze": finetune_params_prefreeze,
        "dataset": dataset_str,
        "tokenizer_path": tokenizer_path,
        "seed": seed,
        "reset_optimizer": reset_optimizer,
        "warmup_steps": warmup_steps,
        "lr_schedule": lr_schedule,
        "warmup_steps_prefreeze": warmup_steps_prefreeze,
        "lr_schedule_prefreeze": lr_schedule_prefreeze,
    }

    lora_configs = {
        "target_modules": args.lora_target_modules,
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout
    }
    if finetune_params_prefreeze == "lora" or finetune_params == "lora":
        params_dict.update(lora_configs)
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_folder = "output"
        hashed_file_name = generate_hashed_dir_name(params_dict, dry_run=dry_run)
        if args.extra_info is not None:
            hashed_file_name = f"{hashed_file_name}_{args.extra_info}"

        if args.experiment_name is not None:
            output_folder = os.path.join(output_folder, args.experiment_name)
            os.makedirs(output_folder, exist_ok=True)
        
        output_dir = os.path.join(output_folder, hashed_file_name)
        logger.info(f"Creating output directory hash using params: {params_dict}")
        logger.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    output_dir_ext = f"final_model"
    final_model_location = os.path.join(output_dir, output_dir_ext)
    if not args.overwrite_final and os.path.exists(final_model_location):
                # raise FileExistsError(f"Final model already exists at {final_model_location}. Use --overwrite-final to proceed.")
        logger.critical(f"Final model already exists at {final_model_location}. Use --overwrite-final to proceed.")
        can_train = False
        save_checkpoints_type = None

    ##### must load in seed:
    try:
        json_load_params = json.load(open(os.path.join(output_dir, "train_config.json")))
        seed = json_load_params["seed"]
        args.seed = seed
        logger.info(f"Loaded seed from output directory: {seed}")
        set_seed(seed)
    except:
        logger.info(f"This should be a new run from scratch, dumping params to output directory")
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(params_dict, f)

    ###### INIT WANDB ######
    if state.is_main_process:
        if args.wandb:
            # TODO Resume?
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

    ###### RESUME FROM CHECKPOINT ######
    # this is to check if this will be a continuation of a previous run or a new run
    checkpoint_path = None
    state_dict = None
    model_freezing_change_occured = False
    current_freeze_params = None # finetune_params_prefreeze if finetune_params_prefreeze is not None else finetune_params

    resume_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoint_path, checkpoints_to_remove = get_latest_checkpoint(output_dir)
            if checkpoint_path is None:
                logger.warning("No checkpoints found to resume from. Starting training from scratch.")
            else:
                if len(checkpoints_to_remove) > 0 and state.is_main_process:
                    logger.info(f"Removing {len(checkpoints_to_remove)} corrupted checkpoints: {checkpoints_to_remove}")
                    remove_old_checkpoints(checkpoints_to_remove, logger)
        else:
            # Load specific checkpoint
            checkpoint_path = os.path.join(output_dir, args.resume_from_checkpoint)
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
            
        logger.info(f"Will attempt to load checkpoint from: {checkpoint_path}")
        if checkpoint_path is not None:
            try:
                logger.info(f"Loading checkpoint metadata from {checkpoint_path}")
                state_dict = torch.load(os.path.join(checkpoint_path, "checkpoint_meta.pt"))
                resume_step = state_dict["update_step"]
                current_freeze_params = state_dict.get("freeze_params", current_freeze_params)

                if resume_step > unfreeze_params_steps and unfreeze_params_steps > 0:
                    logger.info(f"This occurred after unfreeze step, so need to load state")
                    # model, model_is_frozen = setup_lora(model, lora_configs, logger)
                    model_freezing_change_occured = True

                logger.info(f"Resuming training from step {resume_step} with freeze params: {current_freeze_params}")
                    
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                logger.warning("Failed to load checkpoint metadata.")
                raise e


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

    if tokenizer_path is not None:
        # try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded HF tokenizer from vocab file: {tokenizer_path}")
        base_tokenizer = AutoTokenizer.from_pretrained(args.original_model)
        logger.info(f"Loaded base tokenizer from model: {args.original_model}")
        # except:
        #     # load tokenizer from vocab file
        #     base_tokenizer = AutoTokenizer.from_pretrained(args.original_model)
        #     vocab_file_path = tokenizer_path
        #     pre_tok_name = args.pre_tok_name
        #     tokenizer = get_tokenizer(vocab_file_path, pre_tok_name=pre_tok_name, old_tokenizer=base_tokenizer)
        #     logger.info(f"Loaded tokenizer from .model file: {vocab_file_path}")
    else:
        # get original_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.original_model)
        logger.info(f"Loaded tokenizer from model: {args.original_model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_tokenizer is not None and base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token


    original_vocab_size = len(base_tokenizer)
    model.config.original_vocab_size = original_vocab_size
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}, model vocab size: {original_vocab_size}")

    assert num_new_tokens == len(tokenizer) - original_vocab_size, f"tokenizer adding different number than num_new tokens: args: {args.num_new_tokens} != tok: {len(tokenizer) - original_vocab_size}"
    # num_new_tokens = len(tokenizer) - model.config.vocab_size
    
    # if args.num_new_tokens > 0:
    #     assert args.num_new_tokens == num_new_tokens, f"tokenizer adding different number than num_new tokens: args: {args.num_new_tokens} != tok: {num_new_tokens}"
    # args.num_new_tokens = num_new_tokens

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
        if model.config.vocab_size == original_vocab_size + num_new_tokens:
            logger.info(f"Model has already extended embeddings with {num_new_tokens} new tokens")
        else:
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
        data_collator = MyPaddingCollatorGeneral(
            tokenizer=tokenizer,
            max_length=args.max_length if hasattr(args, 'max_length') else None
        )
        other_loss_types = []
        if "mixed" not in eval_other_loss_types:
            eval_other_loss_types.append("mixed")
        if "new_tokens" not in eval_other_loss_types:
            eval_other_loss_types.append("new_tokens")
        if "all" not in eval_other_loss_types:
            eval_other_loss_types.append("all")
        # we will select the loss type based on the dataset row
        # This collator will handle both normal and repeat samples in the same batch.
    else:
        raise ValueError(f"Invalid task name: {task_name}")
    
    logger.info(f"Task: {task_name}, Primary loss: {main_loss_type}, Train tracked losses: {other_loss_types}, Eval tracked losses: {eval_other_loss_types}")

    ###### LOAD DATA ######
    logger.info(f"Loading data from {dataset_str}...")

    # dataset
    dataset_list = dataset_str.split(",")
    if len(dataset_list) > 1:
        ds = load_mixed_dataset(dataset_list, dataset_dir=args.dataset_dir, task_list_split=args.task_list_split)
    else:
        ds = load_from_disk(os.path.join(args.dataset_dir, dataset_str))
    ds = ds.train_test_split(test_size=0.1) # Split the dataset into train (90%) and validation (10%)

    train_loader, train_sampler = create_memory_efficient_loader(
        ds["train"],
        batch_size=batch_size,
        collate_fn=data_collator,
        num_proc=args.num_proc, 
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
    )
    eval_samples = len(ds["test"])
    eval_batches = len(eval_loader)
    eval_loader_token_count = sum(ds["test"]["num_tokens"])
        
    logger.info("Loaded data into dataset")

    ###### TRACK TRAINING STEPS ######
    if args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
    else:
        max_train_steps = train_batches//(gradient_accumulation_steps * num_processes)

    if dry_run and max_train_steps > 50:
        max_train_steps = min(max_train_steps, 10)

    ###### FINETUNING MODES (FREEZING PARAMS) ######
    def manage_model_freezing(model: torch.nn.Module, finetuning_params: str, num_new_tokens: int = None, lora_configs: dict = None):
        # Todo make it so you can unfreeze SOME layers
        model_is_frozen = False
        if finetuning_params == "full":
            model = unfreeze_model(model)
            model_is_frozen = False
        elif finetuning_params == "new_tokens_only":
            model = freeze_old_embeddings(model, num_new_tokens)
            model_is_frozen = True
        elif finetuning_params == "embeddings":
            model = freeze_model_except_embeddings(model)
            model_is_frozen = True
        elif finetuning_params == "lora":
        
            # By default, apply LoRA to all linear layers in attention blocks.
            # Or pick a subset like ["q_proj","v_proj"] if using a Llama-based model.
            if lora_configs is None:
                lora_configs = {}
            model, model_is_frozen = setup_lora(model, lora_configs, logger)
            
        elif finetuning_params == "first_last":
            model = freeze_model_except_embeddings(model)

            # Unfreeze first and last transformer layer
            model = unfreeze_first_last_layer(model, logger)
            model_is_frozen = True
        else:
            raise ValueError(f"Invalid finetuning mode: {finetuning_params}")
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Finetuning_params: {finetuning_params}, Trainable parameters: {trainable_params:,} Total parameters: {total_params:,} ({trainable_params/total_params:.2%} of total)")
        # if args.wandb and state.is_main_process:
        #     wandb.log({"trainable_params": trainable_params, "total_params": total_params})
        
        return model, model_is_frozen
    
    if will_unfreeze_params and not model_freezing_change_occured:
        current_freeze_params = finetune_params_prefreeze
        current_warmup_steps = warmup_steps_prefreeze
        current_lr_schedule = lr_schedule_prefreeze
        current_max_train_steps = unfreeze_params_steps
    else:
        current_freeze_params = finetune_params
        current_warmup_steps = warmup_steps
        current_lr_schedule = lr_schedule
        current_max_train_steps = max_train_steps - unfreeze_params_steps

    model, model_is_frozen = manage_model_freezing(model, current_freeze_params, num_new_tokens, lora_configs)

    logger.info("Creating objects from scratch.  Will load checkpoint if specified")
    ###### INIT OPTIMIZER ######
    def init_optimizer(raw_model, lr: float):
        logger.info(f"Initializing optimizer...")
        optim = torch.optim.AdamW(raw_model.parameters(), lr=lr, fused=True)
        # optim = torch.optim.AdamW((p for p in raw_model.parameters() if p.requires_grad), lr=lr, fused=True)
        logger.info(f"Initial learning rate from optimizer: {optim.param_groups[0]['lr']}")
        return optim
    
    optim = init_optimizer(model, learning_rate)

    ###### INIT SCHEDULER ######
    def init_scheduler(optim, max_train_steps, warmup_steps, lr_schedule: str, current_step: int = -1):
        logger.info(f"Initializing scheduler...")
        if lr_schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim, 
                num_training_steps=max_train_steps,
                num_warmup_steps=warmup_steps,
                last_epoch=current_step
            )
        elif lr_schedule == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optim, 
                num_warmup_steps=warmup_steps,
                # last_epoch=current_step
            )
        elif lr_schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optim, 
                num_training_steps=max_train_steps,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Invalid lr_schedule: {lr_schedule}")
        return scheduler


    scheduler = init_scheduler(optim, 
                               current_max_train_steps,
                               current_warmup_steps, 
                               current_lr_schedule
                               )
    split_scheduler = False  # split schedule will split learning rate across multiple devices by accelerator

    ####### INITIALIZE ACCELERATOR ######
    logger.info(f"Initializing accelerator...")
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    project_config = ProjectConfiguration(project_dir=output_dir,
                                          automatic_checkpoint_naming=True
                                          )
    
    dataloader_config = DataLoaderConfiguration(
        # use_stateful_dataloader=True,
        use_seedable_sampler = True
    )

    # fsdp
    # if args.fsdp:
    #     fsdp_plugin = FullyShardedDataParallelPlugin(
    #         state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    #         optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    #     )
    #     # dataloader_config = None
    #     accelerator = Accelerator(
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         mixed_precision="bf16",
    #         # mixed_precision="fp16",
    #         kwargs_handlers=[timeout],
    #         project_config=project_config,
    #         log_with="wandb" if args.wandb else None,
    #         dataloader_config=dataloader_config,
    #         fsdp_plugin=fsdp_plugin,
    #     )
    # else:
    if True:
        # dataloader_config = None
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="bf16",
            # mixed_precision="fp16",
            kwargs_handlers=[timeout],
            project_config=project_config,
            log_with="wandb" if args.wandb else None,
            dataloader_config=dataloader_config
        )

    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "efficient-tokenization",
    )

    ###### PREPARE ARTIFACTS TO BE USED BY ACCELERATOR ######
    def prepare_all_artifacts(model, optim, scheduler, accelerator = None, train_loader = None, eval_loader = None, split_scheduler: bool = False):
        logger.info(f"Preparing artifacts to be used by accelerator...")
        # prepare artifacts - accelerator handles device placement and dataloader splitting
        model, optim = accelerator.prepare(model, optim)
        if train_loader is not None:
            train_loader = accelerator.prepare_data_loader(train_loader, device_placement=True)
        if eval_loader is not None:
            eval_loader = accelerator.prepare_data_loader(eval_loader, device_placement=True)

        if split_scheduler:
            scheduler = accelerator.prepare(scheduler)
        else:
            accelerator.register_for_checkpointing(scheduler)

        return model, optim, scheduler, train_loader, eval_loader
    
    model, optim, scheduler, train_loader, eval_loader = prepare_all_artifacts(model, optim, scheduler, accelerator, train_loader, eval_loader, split_scheduler)

    # these calculations are done AFTER the accelerator is initialized
    num_batches_per_device_per_epoch = len(train_loader)
    grad_updates_per_device_per_epoch = math.ceil(num_batches_per_device_per_epoch / gradient_accumulation_steps)

    # TODO make is so you can specify num_epochs
    # if num_epochs > 0:
    #     total_gradient_updates = min(max_train_steps, grad_updates_per_device_per_epoch * num_epochs * num_processes)
    #     total_gradient_updates_per_device = min(math.ceil(max_train_steps // num_processes), grad_updates_per_device_per_epoch * num_epochs)
    # else:

    total_gradient_updates = max_train_steps
    # total_gradient_updates_per_device = math.ceil(max_train_steps // num_processes)

    ###### INIT TRAINING STATE ######
    # samples tracking
    epoch = 0
    epoch_step = 0
    update_step = 0  # this was already set but will reset later when loading checkpoint
    total_batched_samples = 0
    prev_training_time = 0

    # loss tracking
    cumulative_batch_counter = 0  # number of batches processed since last accumulation
    cumulative_token_counter = 0  # all tokens processed across all processes
    cumulative_new_token_counter = 0  # all new tokens processed across all processes

    ###### LOAD CHECKPOINT ######
    if checkpoint_path is not None and state_dict is not None:
        # get the current step before loading the checkpoint
        try:
            # Load the state
            logger.info(f"Loading accelerator state from {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            checkpoint_no = int(checkpoint_path.split("_")[-1]) + 1
            accelerator.project_configuration.iteration = checkpoint_no
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.warning("Failed to load checkpoint into accelerator.")
            raise e

    ###### LOAD CHECKPOINT METADATA ######
    skip_iterator = None
    if state_dict is not None:
        # load training state
        try:
            update_step = state_dict["update_step"]  # should already be loaded
            epoch = state_dict["epoch"]
            epoch_step = state_dict["epoch_step"] + 1
            total_batched_samples = state_dict["total_batched_samples"]
            cumulative_batch_counter = state_dict["cumulative_batch_counter"]
            cumulative_token_counter = state_dict["cumulative_token_counter"]
            cumulative_new_token_counter = state_dict["cumulative_new_token_counter"]
            batch_skips = int((epoch_step - 1)*gradient_accumulation_steps)
            prev_training_time = state_dict.get("training_time", 0)
            # batch_skips = len(train_loader) - 5
            # TODO switch to correct finetuning params based on if past unfreeze step
            if batch_skips >= len(train_loader):
                logger.warning(f"Epoch step {batch_skips} is greater than the number of batches in the train loader {len(train_loader)}")
                epoch_step = -1
            else:
                logger.info(f"Skipping {batch_skips} batches (per loader)")
                skip_loader = accelerator.skip_first_batches(train_loader, batch_skips)  # skip step in epoch
                skip_iterator = iter(skip_loader)
            logger.info(f"Resuming training at step {update_step} (epoch {epoch}) (training time {time.strftime('%H:%M:%S', time.gmtime(prev_training_time))})")
            
        except Exception as e:
            logger.info(f"Failed to load checkpoint metadata: {checkpoint_path} resetting training state")
            logger.info(f"Error: {e}")
            raise e

    logger.debug(f"Accelerator state: {accelerator.state}")
    training_iterator = iter(train_loader)
    eval_iterator = iter(eval_loader)

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
        "eval_iters": args.eval_iters,
        "experiment": args.experiment_name,
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
    logger.info(f"Can train: {can_train}")
    logger.info(f"max train steps {max_train_steps}") # ({math.ceil(max_train_steps / num_processes)} per device)")
    logger.info(f"total gradient updates {total_gradient_updates} (per epoch {len(train_loader)/gradient_accumulation_steps})") # ({total_gradient_updates_per_device} per device)")
    logger.info(f"gradient steps {gradient_accumulation_steps}")
    logger.info(f"per device batch size {batch_size}, eval batch size {args.eval_batch_size}")
    logger.info(f"num gpus: {num_processes}")
    logger.info(f"effective batch size {total_batch_size}")
    logger.info(f"learning rate {learning_rate}, schedule_max_steps {max_train_steps}, split_scheduler {split_scheduler}, (unfreeze_params_steps {unfreeze_params_steps})")
    logger.info(f"finetune params prefreeze: {finetune_params_prefreeze}, lr_schedule_prefreeze {lr_schedule_prefreeze}, lr_schedule_prefreeze_steps {unfreeze_params_steps}, warmup_steps_prefreeze {warmup_steps_prefreeze}")
    logger.info(f"finetune params after unfreeze: {finetune_params}, lr_schedule {lr_schedule}, lr_schedule_steps {max_train_steps - unfreeze_params_steps}, warmup_steps {warmup_steps}")
    logger.info(f"checkpointing steps {args.checkpointing_steps}")
    logger.info(f"eval steps: {args.eval_steps}, eval iters: {args.eval_iters}, benchmark steps: {args.benchmark_steps}")
    logger.info(f"Training until {max_train_steps} steps")

    logger.info(f"train samples: {train_samples}, batches: {train_batches}, and tokens: {train_loader_token_count}")
    logger.info(f"train samples per device: {len(train_loader) * batch_size}, batches per device: {len(train_loader)}, and tokens (approx): {train_loader_token_count / num_processes}")
    logger.info(f"eval samples: {eval_samples}, batches: {eval_batches}, and tokens: {eval_loader_token_count}")
    logger.info(f"eval samples per device: {len(eval_loader) * args.eval_batch_size}, batches per device: {len(eval_loader)}, and tokens (approx): {eval_loader_token_count / num_processes}")
    logger.info(f"Task: {task_name}, extending params: {embedding_init_strategy}")
    
    logger.info(f"accelerator distributed type {accelerator.distributed_type}, num_processes {num_processes} on {torch.cuda.get_device_name()}")
    logger.info(f"CPUs: {args.num_proc}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
    logger.info(f"--------------------------------")

    ###### TRAINING LOOP ######
    accelerator.wait_for_everyone()
    logger.info(f"Starting training loop at step {update_step}...")
    logger.info("")
    start_time = time.time()
    training_time = 0
    use_progress_bar = True
    eval_at_start = True
    log_step = args.log_step + update_step  # start logging at the step specified by the log_step arg + the step number of the checkpoint
    
    progress_bar = tqdm(
        range(total_gradient_updates),
        disable=(not use_progress_bar) or (not accelerator.is_local_main_process),  # turning off progress bar
        initial=update_step,
    )

    largest_batch_per_device = 0
    start_accumulating = True
    while update_step < total_gradient_updates:
        loop_start_time = time.time()
        if start_accumulating:
            i = 0
            # moved eval calculation here so that it can happen before the first training step
            will_eval = (update_step % args.eval_steps == 0) and (update_step > 0 or eval_at_start)
            will_benchmark =(update_step % args.benchmark_steps == 0) and (update_step > 0 or eval_at_start) and eval_config.get("run_lm_eval", False)
            
            # eval and benchmark loops
            if will_eval or will_benchmark:
                model.eval()
                logger.debug(f"EVAL:Step {update_step}: Starting evaluation")
                mem_before = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory before running evaluation - Device {accelerator.process_index}: {mem_before:.2f} MB", main_process_only=False)
                move_optimizer_to_cpu(optim)
                torch.cuda.empty_cache()
                results = evaluate(model, eval_iterator, eval_loader, accelerator, eval_config=eval_config, will_benchmark=will_benchmark, will_eval=will_eval)
                move_optimizer_to_gpu(optim)
                model.train()
                torch.cuda.empty_cache()
                gc.collect()
                if accelerator.is_main_process:
                    results = {f"eval_{k}": v for k, v in results.items()}
                    results["step"] = update_step
                    if args.wandb:
                        wandb.log(results)
                    logger.debug(results)
                accelerator.wait_for_everyone()

                mem_after = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory after running evaluation - Device {accelerator.process_index}: {mem_after:.2f} MB", main_process_only=False)

            # TODO add early stopping
            if not can_train:
                logger.info("stopping early because can_train is False")
                break

            logger.debug(f"Starting step {update_step} of {total_gradient_updates}")

            mem_before = torch.cuda.memory_allocated() / 1024**2
            logger.debug(f"Memory before loading batches - Device {accelerator.process_index}: {mem_before:.2f} MB", main_process_only=False)

            # TODO add time of each iteration, and memory usage
            # Changing model freezing parameters
            if finetune_params is not None and unfreeze_params_steps > 0 and update_step >= unfreeze_params_steps and not model_freezing_change_occured:
                logger.info(f"Changing model frozen parameters at step {update_step} from {finetune_params_prefreeze} to {finetune_params}")
                accelerator.wait_for_everyone()
                # 0) save checkpoint before transitioning to new unfreezing params
                if save_checkpoints_type is not None and isinstance(args.checkpointing_steps, int) and update_step > 0:
                    logger.info(f"Checkpointing at update step {update_step}")
                    
                    state_dict = {
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "update_step": update_step,
                        "total_batched_samples": total_batched_samples,
                        "cumulative_batch_counter": cumulative_batch_counter,
                        "cumulative_token_counter": cumulative_token_counter,
                        "cumulative_new_token_counter": cumulative_new_token_counter,
                        "current_batch_size": batch_size, # used to skip first batches
                        "freeze_params": current_freeze_params,
                    }
                    save_checkpoint(accelerator, output_dir, state_dict, logger, delete_old_checkpoints=not save_checkpoints_type == "all", special_save_location_name = "unfreeze_params")  # delete old checkpoints if not saving all

                # 1) unwrap model
                unwrapped_model = accelerator.unwrap_model(model)
                model = release_memory(model)
                accelerator._models.clear()

                # 2) unfreeze some layers, or wrap with LoRA, etc.
                unwrapped_model, model_is_frozen = manage_model_freezing(unwrapped_model, finetune_params_prefreeze, num_new_tokens, lora_configs)
                continue_optimizer = False
                if continue_optimizer:
                    # TODO this doesnt work and im not sure if we should even use it?
                    old_lr = optim.param_groups[0]["lr"]
                    optim = release_memory(optim)
                    scheduler = release_memory(scheduler)

                    accelerator._optimizers.clear()
                    accelerator._custom_objects.clear()  # TODO hacky
                    # 3) create new optimizer
                    optim = init_optimizer(unwrapped_model, lr = old_lr)
                    # 4) create or re-init your scheduler with new_optimizer
                    scheduler = init_scheduler(optim, max_train_steps - update_step, warmup_steps, lr_schedule, current_step=update_step)
                    # 5) re-prepare these artifacts
                    model, optim, scheduler, _, _ = prepare_all_artifacts(unwrapped_model, optim, scheduler, accelerator, train_loader = None, eval_loader = None, split_scheduler = split_scheduler)
                elif reset_optimizer:
                    optim = release_memory(optim)
                    scheduler = release_memory(scheduler)

                    accelerator._optimizers.clear()
                    accelerator._custom_objects.clear()  # TODO hacky
                    # 3) create new optimizer
                    optim = init_optimizer(unwrapped_model, lr = learning_rate)
                    # 4) create or re-init your scheduler with new_optimizer
                    scheduler = init_scheduler(optim, max_train_steps - update_step, warmup_steps, lr_schedule, current_step=-1)
                    # 5) re-prepare these artifacts
                    model, optim, scheduler, _, _ = prepare_all_artifacts(unwrapped_model, optim, scheduler, accelerator, train_loader = None, eval_loader = None, split_scheduler = split_scheduler)
                else:
                    model = accelerator.prepare(unwrapped_model)

                model_freezing_change_occured = True
                current_freeze_params = finetune_params
                
            loss_log = None
            grad_norm = None
            new_embeddings_grad_norm = None
            old_embeddings_grad_norm = None
            new_embeddings_norm = None
            old_embeddings_norm = None
            
            # num_batches_in_step = gradient_accumulation_steps if epoch_step != (grad_updates_per_device_per_epoch - 1) else epoch_remainder_on_device
            logger.debug(f"gradient accumulation steps: {gradient_accumulation_steps} total gradient updates: {total_gradient_updates}")

            # get local num items in batch
            device_memory_usage = torch.cuda.memory_allocated() / 1024**2
            logger.debug(f"Step {update_step} - Device {accelerator.process_index} - device memory usage {device_memory_usage} MB, largest_batch_per_device {largest_batch_per_device}", main_process_only=False)

            accumulation_batch_counter = 0
            new_token_counter = 0
            total_token_counter = 0
            accumulated_losses_per_loss_type = defaultdict(list)
            accumulated_tokens_per_loss_type = defaultdict(list)
            model.train()
            start_accumulating = False
        
        batch, training_iterator, epoch, epoch_step, skip_iterator = get_next_batch(training_iterator, train_loader, epoch, epoch_step, skip_iterator)
        i += 1
        total_batched_samples += 1

        # Move batch to device just before use
        batch = {k: v.to(accelerator.device) if k != "task_type" else v for k, v in batch.items()}
        mem_before_forward = torch.cuda.memory_allocated() / 1024**2
        largest_batch_per_device = max(largest_batch_per_device, batch['input_ids'].numel())
        num_items_in_batch = (batch["labels"].ne(-100)).sum(1).cpu().tolist()
        max_id = batch["input_ids"].max()

        logger.debug(f"Memory before forward pass opt_step: {update_step} accum_step:{i} - Device {accelerator.process_index}: {mem_before_forward:.2f} MB, "
                        f"largest_batch_per_device {largest_batch_per_device} "
                        f"ids: {batch['input_ids'].numel()}, tokens with losses: {num_items_in_batch}, max_id: {max_id}", main_process_only=False
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

            del main_loss, tracked_losses, num_items_for_loss, tracked_tokens_per_loss_type, batch
            # torch.cuda.empty_cache()  # Consider moving this outside the loop
            
            if accelerator.sync_gradients:
                if num_new_tokens > 0:
                    new_embeddings_list = get_new_embedding_params(model, num_new_tokens)
                    new_embeddings_norm = calculate_norm(new_embeddings_list)

                    new_embeddings_grad_list = get_new_embeddings_grads(model, num_new_tokens)
                    new_embeddings_grad_norm = calculate_norm(new_embeddings_grad_list)
                    
                    # Also compute norms (not gradients) for new/old embeddings
                    del new_embeddings_list, new_embeddings_grad_list

                old_embeddings_list = get_old_embedding_params(model, num_new_tokens)
                old_embeddings_norm = calculate_norm(old_embeddings_list)

                old_embeddings_grad_list = get_old_embeddings_grads(model, num_new_tokens)
                old_embeddings_grad_norm = calculate_norm(old_embeddings_grad_list)
                del old_embeddings_list, old_embeddings_grad_list

                if args.grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_norm)

                optim.step()
                # # TODO look at single neuron and gradient
                logger.debug(f"Device {accelerator.process_index} - Step {update_step}, {accumulation_batch_counter}: {scheduler.get_last_lr()}")
                scheduler.step()
                
                optim.zero_grad()
                memory_after_grad_step = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory after grad step - Device {accelerator.process_index}: {memory_after_grad_step:.2f} MB", main_process_only=False)
    
        # torch.cuda.empty_cache()
                
        if accelerator.sync_gradients:
            # SYNC POINT
            memory_before_gather = torch.cuda.memory_allocated() / 1024**2
            logger.debug(f"SYNC POINT - Steps {update_step}, "
                        f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}, "
                        f"Gradient state steps: {accelerator.gradient_state.num_steps}, "
                        f"Our counter: {accumulation_batch_counter}, "
                        f"Memory before gather: {memory_before_gather:.2f} MB")

            gathered_metrics_list = [
                accumulation_batch_counter,
                new_token_counter,
                total_token_counter,
                accumulated_losses_per_loss_type,
                accumulated_tokens_per_loss_type,
            ]

            accelerator.wait_for_everyone()  # this is probably not needed as it happens inside gather_for_metrics
            
            # All processes must participate in gather
            gathered_metrics = accelerator.gather_for_metrics(gathered_metrics_list)

            log_time = time.time()
            loop_time = log_time - loop_start_time
            training_time = (log_time - start_time) + prev_training_time

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
                    "old_embeddings_grad_norm": old_embeddings_grad_norm,
                    "new_embeddings_norm": new_embeddings_norm,
                    "old_embeddings_norm": old_embeddings_norm,
                    "loop_time": loop_time,
                }
                metrics_dict = {f"train_{k}": v for k, v in metrics_dict.items()}
                metrics_dict["step"] = update_step
                metrics_dict["log_step"] = log_step
                metrics_dict["epoch"] = epoch
                metrics_dict["cumulative_training_time"] = training_time
                loss_log = {k: v for k, v in losses_dict.items() if not k.endswith("_weighted_loss")}
            
                # log_loss(metrics_dict, output_dir)
                if args.wandb:
                    wandb.log(metrics_dict)
            
                # only update progress bar on main process also only do checkpointing on main process
                if loss_log is not None:
                    if use_progress_bar:
                        loss_log["loop_time"] = time.strftime("%H:%M:%S", time.gmtime(loop_time))
                        progress_bar.set_postfix(loss_log)  # This will now show both raw loss and moving average
                    else:
                        logger.info(metrics_dict)
                
                mem_after_gather = torch.cuda.memory_allocated() / 1024**2
                logger.debug(f"Memory after gather - Device {accelerator.process_index}: {mem_after_gather:.2f} MB", main_process_only=False)

                del gathered_metrics
                del metrics_dict
                del all_gathered_losses
                del all_gathered_tokens
                del losses_dict
                del loss_log
        
            del gathered_metrics_list
            del new_embeddings_grad_norm
            del old_embeddings_grad_norm
            del new_embeddings_norm
            del old_embeddings_norm

            # gc.collect()
            memory_after_cleanup = torch.cuda.memory_allocated() / 1024**2
            logger.debug(f"Memory after cleanup - Device {accelerator.process_index}: {memory_after_cleanup:.2f} MB", main_process_only=False)

            # checkpointing (outsize main process)
            if save_checkpoints_type is not None and isinstance(args.checkpointing_steps, int) and update_step > 0:
                if update_step % args.checkpointing_steps == 0:
                    logger.info(f"Checkpointing at update step {update_step}")
                    
                    state_dict = {
                        "update_step": update_step,
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "log_step": log_step,
                        "total_batched_samples": total_batched_samples,
                        "cumulative_batch_counter": cumulative_batch_counter,
                        "cumulative_token_counter": cumulative_token_counter,
                        "cumulative_new_token_counter": cumulative_new_token_counter,
                        "current_batch_size": batch_size, # used to skip first batches
                        "freeze_params": current_freeze_params,
                        "training_time": training_time,
                    }
                    save_checkpoint(accelerator, output_dir, state_dict, logger, delete_old_checkpoints=not save_checkpoints_type == "all")  # delete old checkpoints if not saving all
    
            #TODO  do early stopping
            logger.debug(f"Updating progress bar - Device {accelerator.process_index}, step {update_step}")

            start_accumulating = True
            i = 1
            epoch_step += 1
            update_step += 1
            log_step += 1
            accumulated_losses_per_loss_type.clear()
            accumulated_tokens_per_loss_type.clear()
            progress_bar.update(1)
    
    accelerator.wait_for_everyone()
    full_training_time = time.time() - start_time
    logger.info(f"Training Finished in {time.strftime('%H:%M:%S', time.gmtime(full_training_time))}")
    progress_bar.close()

    training_state_dict = {
            "update_step": update_step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            "log_step": log_step,
            "total_batched_samples": total_batched_samples,
            "cumulative_batch_counter": cumulative_batch_counter,
            "cumulative_token_counter": cumulative_token_counter,
            "cumulative_new_token_counter": cumulative_new_token_counter,
            "current_batch_size": batch_size,  # used to skip first batches
            "freeze_params": current_freeze_params,
            "training_time": full_training_time,
        }
    
    if update_step < total_gradient_updates:
        logger.info(f"Training finished before total gradient updates {total_gradient_updates} were reached, only got to {update_step}")
        if save_checkpoints_type is not None:
            save_checkpoints_type = "all"  # dont delete old checkpoints
    
    # Make sure all processes are synced
    output_dir_path = os.path.join(output_dir, output_dir_ext)
    if save_checkpoints_type is not None:
        logger.info(f"Preparing to save model to {output_dir}")
        
        if not check_disk_space(output_dir, required_space_gb=20):
            logger.info("Aborting save due to insufficient disk space")
            return

        unwrapped_model = accelerator.unwrap_model(model)
        if isinstance(unwrapped_model, PeftModel):
            # merge the LoRA weights into the backbone
            unwrapped_model = unwrapped_model.merge_and_unload()  # merges LoRA offsets into the backbone
        
        if accelerator.is_main_process:
            save_training_state_dict(output_dir_path, training_state_dict, logger)

        unwrapped_model.save_pretrained(
            output_dir_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            # state_dict=state_dict,
        )
        if save_checkpoints_type == "model_only":
            if accelerator.is_main_process:
                remove_old_checkpoints(output_dir, logger, "checkpoints")
        else:
            save_checkpoint(accelerator, output_dir, training_state_dict, logger, delete_old_checkpoints=not save_checkpoints_type == "all")

    # Clear memory before final evaluation
    accelerator.clear()
    torch.cuda.empty_cache()
    gc.collect()
    del unwrapped_model

    # Move optimizer to CPU and clear GPU memory
    move_optimizer_to_cpu(optim)
    torch.cuda.empty_cache()
    
    # Run final evaluation
    eval_metrics_dict = evaluate(model, eval_iterator, eval_loader, accelerator, eval_config=eval_config, will_benchmark=eval_config.get("run_lm_eval", True), will_eval=True)

    # move_optimizer_to_gpu(optim)
    if accelerator.is_main_process:
        log_dict = {k: v for k, v in eval_metrics_dict.items() if not k.endswith("_weighted_loss")}
        logger.debug(f"EVAL:Step {update_step}: Eval Loss = {log_dict[f'{main_loss_type}_loss']:.4f}")

        eval_metrics_dict = {f"eval_{k}": v for k, v in eval_metrics_dict.items()}
        eval_metrics_dict["step"] = update_step

        # Use the same dict for both logging and wandb
        if args.wandb:
            wandb.log(eval_metrics_dict)
        
        logger.info(f"Saving Finished")

    if accelerator.is_main_process:
        lm_eval_string = get_lm_eval_string(output_dir_path, 
                                            tokenizer.name_or_path, 
                                            tasks=eval_config["benchmark_tasks"],
                                            # num_fewshot=eval_config["num_fewshot"],
                                            limit=eval_config["limit"],
                                            log_samples=eval_config["log_samples"],
                                            # cache_requests=eval_config["cache_requests"],
                                            # show_config=eval_config["show_config"],
                                            experiment = args.experiment_name,
                                            )
        
        with open(os.path.join(output_dir, "lm_eval.sh"), "w") as f:
            f.write(lm_eval_string)

    # Clear everything before ending training
    del model
    del optim
    del eval_loader
    del train_loader
    gc.collect()

    # Make sure all processes are synced before ending training
    logger.info("COMPLETED TRAINING")
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
