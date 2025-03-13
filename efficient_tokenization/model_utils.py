import torch
from typing import Dict, List, Tuple
import os
import shutil

def save_checkpoint(accelerator, output_dir, state_dict, logger):
    # Delete previous checkpoints before saving new one
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        checkpoint_prefix = "checkpoint"
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith(checkpoint_prefix)]
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            logger.debug(f"Removing old checkpoint: {checkpoint_path}", main_process_only=False)
            try:
                shutil.rmtree(checkpoint_path)
            except Exception as e:
                logger.warning(f"Error removing checkpoint {checkpoint_path}: {e}")
    accelerator.wait_for_everyone()
    save_location = accelerator.save_state()  # accelerator handles state name

    torch.save(state_dict, os.path.join(save_location, "checkpoint_meta.pt"))

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


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def calc_loss_without_grad(model, batch: Dict[str, torch.Tensor], new_tokens_mask: torch.Tensor, loss_types: List[str], materialize_logits: bool = True) -> Tuple[Dict[str, float], Dict[str, int]]:
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
                if "loss_mask" not in batch:
                    raise ValueError("loss_mask not in batch")
                labels = batch["labels"].clone()
                labels[batch["loss_mask"] == 0] = -100
            elif loss_type == "new_tokens":
                labels = batch["labels"].clone()
                labels[new_tokens_mask] = -100
            else:
                raise ValueError(f"Invalid loss_type: {loss_type}")
            
            if materialize_logits:
                # Calculate the loss using the model's loss function
                loss = loss_function(logits, labels, vocab_size=model.module.config.vocab_size)
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels, use_cache=False)
                loss = outputs.loss
            
            # Store the loss in the dictionary
            losses[loss_type] = loss.item()
            num_tokens_per_loss_type[loss_type] = (labels.ne(-100)).sum().item()

    return losses, num_tokens_per_loss_type


def forward_pass(model, batch: Dict[str, torch.Tensor], loss_with_grad: str = "all", losses_without_grad: List[str] = [], materialize_logits: bool = True) -> Tuple[torch.Tensor, Dict[str, float], int, Dict[str, int]]:
    # THIS IS CALLED INSIDE accelerator.accumulate()
    # losses_without_grad can be "all", "translated", "new_tokens"
    original_vocab_size = getattr(model.module.config, "original_vocab_size", 0)
    
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
            labels = batch["labels"].clone()
            labels[batch["loss_mask"] == 0] = -100
        elif loss_with_grad == "new_tokens":
            labels = batch["labels"].clone()
            labels[new_tokens_mask] = -100
        else:
            raise ValueError(f"Invalid loss_with_grad: {loss_with_grad}")
        # num_items_in_batch=num_items_in_batch, 
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels, use_cache=False)
        main_loss = outputs.loss

        num_items_for_loss = (labels.ne(-100)).sum().item()

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
