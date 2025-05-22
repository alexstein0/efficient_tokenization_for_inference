import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import numpy as np
from datasets import concatenate_datasets, load_from_disk, Dataset, load_dataset, DatasetDict
from collections import defaultdict
import os


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

        # replace_labels = False # TODO: implement this
        # if replace_labels:
        #     loss_mask_list = []
        #     # only calculate loss on the tokens that are not masked
        #     for label, loss_mask_item in zip(labels, loss_mask):
        #         label[loss_mask_item==0] = -100
        #         loss_mask_list.append(label)

        #     loss_mask = loss_mask_list
        #     del loss_mask_list

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


@dataclass
class MyPaddingCollatorGeneral:
    """
    Data collator that will dynamically pad the inputs received.
    If a sample has "task_type = 'repeat'", we mask out the prefix portion of labels.
    Otherwise (task_type='normal'), we do standard next-token labeling.
    """
    tokenizer: Any
    max_length: int = None
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract all input_ids, attention_masks, labels
        # plus optional loss_mask, and optional task_type
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # If no "loss_mask" in features, define them as all ones
        if "loss_mask" in features[0]:
            loss_mask = [f["loss_mask"] for f in features]
        else:
            loss_mask = [None]*len(features)

        # Check if we have "task_type" in each sample
        # We'll store 'task_type' as a list if present, else None
        has_task_type = ("task_type" in features[0])
        task_types = [f["task_type"] if has_task_type and "task_type" in f else None for f in features]

        # Convert all to torch tensors if not already
        if isinstance(input_ids[0], list):
            input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        if isinstance(attention_mask[0], list):
            attention_mask = [torch.tensor(x, dtype=torch.long) for x in attention_mask]
        if isinstance(labels[0], list):
            labels = [torch.tensor(x, dtype=torch.long) for x in labels]
        for idx in range(len(loss_mask)):
            if loss_mask[idx] is None:
                # Create a default mask of all 1's the same length as input_ids[idx]
                loss_mask[idx] = torch.ones(len(input_ids[idx]), dtype=torch.long)
            elif isinstance(loss_mask[idx], list):
                loss_mask[idx] = torch.tensor(loss_mask[idx], dtype=torch.long)

        # # If a sample is 'repeat', mask out prefix tokens in 'labels'
        # # For example, define the prefix as half the tokens. 
        # for i, (lab, mask_type) in enumerate(zip(labels, task_types)):
        #     if mask_type == "repeat":
        #         prefix_len = lab.size(0) // 2  # or your own logic
        #         # Set the prefix portion to -100 so no loss is computed there
        #         lab[:prefix_len] = -100

        # Compute the maximum length for padding
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

        for ids, attn, lab, lmask in zip(input_ids, attention_mask, labels, loss_mask):
            pad_len = max_length - ids.size(0)

            if pad_len > 0:
                # Pad input_ids
                padded_input = torch.cat([
                    ids,
                    torch.ones(pad_len, dtype=torch.long) * self.tokenizer.pad_token_id
                ])
                # Pad attention_mask
                padded_attn = torch.cat([
                    attn,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
                # Pad labels
                padded_lab = torch.cat([
                    lab,
                    torch.ones(pad_len, dtype=torch.long) * -100
                ])
                # Pad loss_mask
                padded_lmask = torch.cat([
                    lmask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            else:
                padded_input = ids
                padded_attn = attn
                padded_lab = lab
                padded_lmask = lmask

            padded_input_ids.append(padded_input)
            padded_attention_mask.append(padded_attn)
            padded_labels.append(padded_lab)
            padded_loss_mask.append(padded_lmask)
        
        # Stack all tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "loss_mask": torch.stack(padded_loss_mask),
            "task_type": task_types
        }
        
        return batch
    
def load_dataset_from_disk_or_hf(dataset_name: str, dataset_dir: str = "datasets") -> DatasetDict | Dataset:
    try:
        ds = load_from_disk(os.path.join(dataset_dir, dataset_name))
    except:
        ds = load_dataset(dataset_name)
    
    if isinstance(ds, DatasetDict):
        return ds
    else:
        return ds.train_test_split(test_size=0.1)
    
    
def load_mixed_dataset(dataset_name_list: List[str], dataset_dir: str = "datasets", task_list_split: str | None = None) -> DatasetDict:
    if task_list_split is None:
        print(f"WARNING: task_list_split is None, using uniform split")
        task_list_split = [1.0/len(dataset_name_list) for _ in dataset_name_list]
    else:
        task_list_split = [float(x) for x in task_list_split.split(",")]
        if len(task_list_split) != len(dataset_name_list):
            print(f"WARNING: task_list_split length {len(task_list_split)} does not match task length {len(dataset_name_list)}, using uniform split")
            task_list_split = [1.0/len(dataset_name_list) for _ in dataset_name_list]
        else:
            total = sum(task_list_split)
            task_list_split = [x/total for x in task_list_split]

    assert len(dataset_name_list) == len(task_list_split)
    assert sum(task_list_split) == 1
    dataset_list = defaultdict(list)
    lengths = {}
    for i, dataset_name in enumerate(dataset_name_list):
        ds = load_dataset_from_disk_or_hf(dataset_name, dataset_dir)
        
        for split in ds.keys():
            # separates for each train and test split
            dataset_list[split].append(ds[split])

            num_samples = ds[split].num_rows
            if i == 0:
                # TODO this doesnt work for datasets of different sizes (such as dictionary datasets)
                lengths[split] = num_samples
            else:
                assert lengths[split] == num_samples, f"All datasets must have the same number of samples, but {lengths[split]} != {num_samples} for dataset {dataset_name} split {split}"
    
    ds_dict = {}
    for split in dataset_list.keys():
        # Create random values between 0 and 1
        dist = np.random.random(lengths[split])
        mixed_ds_list = []
        # Calculate cumulative percentages for proper binning
        cum_pcts = np.cumsum([0] + task_list_split)
        
        # Assign each random value to a bin
        for i in range(len(task_list_split)):
            # Select indices where random values fall between current and next cumulative percentage
            idx = np.where((dist >= cum_pcts[i]) & (dist < cum_pcts[i+1]))[0]
            mixed_ds_list.append(dataset_list[split][i].select(idx))

        ds_dict[split] = concatenate_datasets(mixed_ds_list)
    return ds_dict


def create_memory_efficient_loader(dataset, batch_size, collate_fn, num_proc, shuffle=True):
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
        prefetch_factor=1,  # Reduce prefetching
        # persistent_workers=True,  # THIS throws an error
        num_workers=1,  # Adjust based on CPU cores  
        # if num workers > 1 this is the reason for the timeout in debugger
        timeout=60
    )
    return loader, sampler
