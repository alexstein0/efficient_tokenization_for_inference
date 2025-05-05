import torch
from typing import Dict, List, Tuple
import os
import shutil
from efficient_tokenization.extend_embeddings import freeze_old_embeddings, freeze_model_except_embeddings, unfreeze_model, unfreeze_embeddings
from peft import get_peft_model, LoraConfig, TaskType

def calc_batch_size_stuff(total_batch_size: int = None, batch_size: int = None, num_processes: int = None, gradient_accumulate_every: int = None):
    if total_batch_size is not None:
        total_batch_size = total_batch_size
        if batch_size is not None:
            batch_size = batch_size
            gradient_accumulate_every = total_batch_size // (batch_size * num_processes)
        elif gradient_accumulate_every is not None:
            gradient_accumulate_every = gradient_accumulate_every
            batch_size = total_batch_size // (gradient_accumulation_steps * num_processes)
        else:
            raise ValueError("Either batch_size or gradient_accumulate_every must be provided if inferring from total_batch_size")
    else:
        if batch_size is None or gradient_accumulate_every is None:
            raise ValueError("Both batch_size and gradient_accumulate_every must be provided total_batch_size not set")

    # make sure the rounding is correct
    total_batch_size = batch_size * num_processes * gradient_accumulate_every
    return total_batch_size, batch_size, gradient_accumulate_every

def move_old_checkpoints_to_temp(output_dir: str, paths_to_delete: List[str], temp_dir_name: str = "temp_checkpoints", new_name: str = None) -> List[str]:
    temp_dir = os.path.join(output_dir, temp_dir_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    all_temp_paths = []
    for path in paths_to_delete:
        if new_name is not None:
            temp_path = os.path.join(temp_dir, new_name)
        else:
            temp_path = os.path.join(temp_dir, os.path.basename(path))
        shutil.move(path, temp_path)
        all_temp_paths.append(temp_path)

    return all_temp_paths

def save_checkpoint(accelerator, output_dir, state_dict, logger, delete_old_checkpoints: bool = True, special_save_location_name: str = None):
        # Delete previous checkpoints before saving new one
    paths_to_delete = []
    if delete_old_checkpoints:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        paths_to_delete = get_checkpoint_paths(checkpoint_dir)
        # logger.info(f"Removed old checkpoints from {output_dir}")
        logger.info(f"Found old checkpoints to delete: {paths_to_delete}")
        if accelerator.is_main_process:
            paths_to_delete = move_old_checkpoints_to_temp(output_dir, paths_to_delete)
    
    # Make sure all processes are synchronized before saving
    accelerator.wait_for_everyone()
    
    # Save the state
    save_location = accelerator.save_state()  # accelerator handles state name
    
    # Make sure all processes are synchronized after save_state
    accelerator.wait_for_everyone()
    
    # Save the metadata
    try:
        if accelerator.is_main_process:
            save_training_state_dict(save_location, state_dict, logger)
            remove_old_checkpoints(paths_to_delete, logger)
    except Exception as e:
        logger.error(f"Error during checkpoint saving process: {str(e)}")
    
    # Final synchronization to ensure all processes complete together
    accelerator.wait_for_everyone()

    if special_save_location_name is not None:
        # if keeping specific training states
        save_location = move_old_checkpoints_to_temp(output_dir, [save_location], temp_dir_name = "kept_training_states", new_name = special_save_location_name)
    
    accelerator.wait_for_everyone()

def save_training_state_dict(save_location, state_dict, logger) -> bool:
    # Must be main process
    try:
        torch.save(state_dict, os.path.join(save_location, "checkpoint_meta.pt"))
        logger.info(f"Checkpoint metadata saved to {save_location}")
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint metadata: {str(e)}")
        return False


def get_checkpoint_paths(checkpoint_dir: str):
    all_checkpoint_paths = []
    if os.path.exists(checkpoint_dir):
        for checkpoint in os.listdir(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            if os.path.exists(checkpoint_path):
                all_checkpoint_paths.append(checkpoint_path)
    return all_checkpoint_paths

def remove_old_checkpoints(output_dir: str | List[str], logger, checkpoint_ext: str = "checkpoints"):
    # MUST BE MAIN PROCESS
    if isinstance(output_dir, str):
        checkpoint_dir = os.path.join(output_dir, checkpoint_ext)
        all_checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    else:
        all_checkpoint_paths = output_dir
    for checkpoint_path in all_checkpoint_paths:
        logger.debug(f"Removing old checkpoint: {checkpoint_path}")
        try:
            shutil.rmtree(checkpoint_path)
        except Exception as e:
            logger.warning(f"Error removing checkpoint {checkpoint_path}: {e}")

def calculate_norm(parameters, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    
    total_norm = torch.nn.utils.get_total_norm(parameters, norm_type, error_if_nonfinite, foreach)
    return total_norm


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    if "embed_tokens" in lora_module_names:
        lora_module_names.remove("embed_tokens")

    return list(lora_module_names)


def calc_loss_without_grad(model, batch: Dict[str, torch.Tensor | List[str]], new_tokens_mask: torch.Tensor, loss_types: List[str], materialize_logits: bool = True) -> Tuple[Dict[str, float], Dict[str, int]]:
    model.eval()
    losses = {}
    num_tokens_per_loss_type = {}
    
    with torch.no_grad():
        if materialize_logits:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            logits = outputs.logits
            loss_function = model.module.loss_function

        for loss_type in loss_types:
            # Determine the labels based on the loss type
            if loss_type == "all": 
                labels = batch["labels"]
            elif loss_type == "translated":
                # if loss_mask is not present, raise an error
                if "loss_mask" not in batch:
                    raise ValueError("loss_mask not in batch")
                labels = batch["labels"].clone()
                labels[batch["loss_mask"] == 0] = -100
            elif loss_type == "new_tokens":
                labels = batch["labels"].clone()
                labels[~new_tokens_mask] = -100
            elif loss_type == "mixed":
                # we will select the loss type based on the dataset row
                # This collator will handle both normal and repeat samples in the same batch.
                labels = batch["labels"].clone()
                labels[batch["loss_mask"] == 0] = -100
                # TODO check if this is correct
            else:
                raise ValueError(f"Invalid loss_type: {loss_type}")
            
            num_tokens_for_loss = (labels.ne(-100)).sum().item()
            if num_tokens_for_loss == 0:
                losses[loss_type] = 0.0
                num_tokens_per_loss_type[loss_type] = 0
                continue

            if materialize_logits:
                # Calculate the loss using the model's loss function
                loss = loss_function(logits, labels, vocab_size=model.module.config.vocab_size)
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels, use_cache=False)
                loss = outputs.loss
            
            # Store the loss in the dictionary
            losses[loss_type] = loss.item()
            num_tokens_per_loss_type[loss_type] = num_tokens_for_loss

    return losses, num_tokens_per_loss_type

def forward_pass(model, 
                 batch: Dict[str, torch.Tensor | List[str]], 
                 loss_with_grad: str = "all", 
                 losses_without_grad: List[str] = [], 
                 materialize_logits: bool = True,
                 ) -> Tuple[torch.Tensor, Dict[str, float], int, Dict[str, int]]:
    
    # THIS IS CALLED INSIDE accelerator.accumulate()
    # losses_without_grad can be "all", "translated", "new_tokens"
    if hasattr(model, "module"):
        original_vocab_size = getattr(model.module.config, "original_vocab_size", 0)
    else:
        original_vocab_size = getattr(model.config, "original_vocab_size", 0)
    
    if original_vocab_size > 0:
        new_tokens_mask = batch["input_ids"] >= original_vocab_size # TODO check > or >=
    else:
        new_tokens_mask = torch.zeros_like(batch["labels"]) # dont mask anything out

    # if "loss_mask" not in batch:
    #     batch["loss_mask"] = torch.ones_like(batch["labels"])

    num_items_in_batch = (batch["labels"].ne(-100)).sum().item()

    # 1. Forward pass
    main_loss = None
    num_items_for_loss = None
    if loss_with_grad is not None:
        if loss_with_grad == "all": 
            labels = batch["labels"]
        elif loss_with_grad == "translated":
            # if loss_mask is not present, raise an error
            if "loss_mask" not in batch:
                raise ValueError("loss_mask not in batch")
            labels = batch["labels"].clone()
            labels[batch["loss_mask"] == 0] = -100
        elif loss_with_grad == "new_tokens":
            labels = batch["labels"].clone()
            labels[~new_tokens_mask] = -100
        elif loss_with_grad == "mixed":
            # we will select the loss type based on the dataset row
            # This collator will handle both normal and repeat samples in the same batch.
            labels = batch["labels"].clone()
            labels[batch["loss_mask"] == 0] = -100
            # TODO check if this is correct
        else:
            raise ValueError(f"Invalid loss_with_grad: {loss_with_grad}")

        num_items_for_loss = (labels.ne(-100)).sum().item()
        
        # TODO manage memory better
        # new_tokens_mask = new_tokens_mask.cpu()
        # batch["labels"] = batch["labels"].cpu()
        # batch["loss_mask"] = batch["loss_mask"].cpu()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels, use_cache=False)
        main_loss = outputs.loss

    if len(losses_without_grad) > 0:
        tracked_losses, tracked_num_tokens = calc_loss_without_grad(model, batch, new_tokens_mask, losses_without_grad, materialize_logits=materialize_logits)
    else:
        tracked_losses = {}
        tracked_num_tokens = {}

    # makes sure these are always tracked
    tracked_num_tokens["new_tokens"] = (new_tokens_mask.sum().item())
    tracked_num_tokens["all"] = num_items_in_batch
    return main_loss, tracked_losses, num_items_for_loss, tracked_num_tokens


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


def setup_lora(model, lora_configs, logger):
    model = freeze_model_except_embeddings(model)

    target_modules_names = lora_configs.get("target_modules", "linear")

    if target_modules_names == "linear":
        target_modules = find_all_linear_names(model)
    elif target_modules_names == "qv":
        target_modules = ["q_proj", "v_proj"]
    else:
        raise ValueError(f"Invalid target modules: {target_modules_names}")
    
    r = lora_configs.get("r", 16)
    lora_alpha = lora_configs.get("lora_alpha", 64)
    lora_dropout = lora_configs.get("lora_dropout", 0.05)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    model = unfreeze_embeddings(model)
    model_is_frozen = True
    logger.info(f"Lora params target_modules: {target_modules}, r: {r}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}")
    
    return model, model_is_frozen