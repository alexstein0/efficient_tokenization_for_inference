import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

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

        replace_labels = False # TODO: implement this
        if replace_labels:
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


def create_memory_efficient_loader(dataset, batch_size, collate_fn, num_proc, accelerator=None, shuffle=True):
    """Create a memory-efficient data loader"""
    # if accelerator is not None and accelerator.num_processes > 1:
    # sampler = DistributedSampler(dataset)
    # else:
    # single GPU
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,  # Reduce prefetching
        # persistent_workers=True,  # THIS throws an error
        num_workers=num_proc  # Adjust based on CPU cores
    )
    return loader, sampler
