import logging

from efficient_tokenization.tokenize_simple import get_tokenized_data, flatten_genqa_conversations, my_tokenize, get_genqa_data, get_tokenizer, create_translation_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
import copy
import torch
import os
import psutil
import argparse

os.environ.pop('TMPDIR', None)

def setup_logging(log_level=logging.INFO):
    """Setup global logging configuration"""
    # Set up basic configuration first
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=log_level,
        force=True  # This ensures we override any existing configuration
    )
    
    # Get the logger
    logger = logging.getLogger(__name__)  # if you want to see for all ranks
    
    return logger
    
def preprocess_data(ds, args):
    if args.raw_data_name == "genqa":
        ds = get_genqa_data(ds, track_role=True, batch_size=args.cpu_batch_size, threads=args.num_proc)
    else:
        raise ValueError(f"Unsupported raw data name: {args.raw_data_name}")
    return ds
    
def main(args):
    logger = setup_logging()
    # Get tokenizer
    logger.info(f"Getting tokenizer")

    base_tokenizer = None
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
            logger.info(f"Loaded tokenizer from .model file: {vocab_file_path} and pre_tok_name: {pre_tok_name}")
    else:
        # get original_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info(f"Loaded tokenizer from model: {args.model}")


    # Get Dataset
    # the data must be fully preprocessed and ready to be tokenized with the tokenized text in "text" column
    dataset_path = args.dataset_path

    logger.info(f"Downloading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)
    changed_dataset = False

    # this block is the only thing that is dataset specific
    ds = preprocess_data(ds, args)
    if args.task == "translation":
        if base_tokenizer is None:
            base_tokenizer = AutoTokenizer.from_pretrained(args.model)
        ds = create_translation_dataset(ds, base_tokenizer, tokenizer, args.cpu_batch_size, args.num_proc)


    # tokenize dataset
    if args.force_tokenize or "input_ids" not in ds.column_names:
        logger.info(f"Tokenizing dataset")
        ds = my_tokenize(ds.select_columns("text"), tokenizer, args.cpu_batch_size, args.num_proc)
        changed_dataset = True
        logger.info(f"Tokenized dataset")
    else:
        logger.info(f"Dataset already tokenized")

    # Ensure dataset is tokenized
    if "input_ids" not in ds.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    
    # add labels
    if "labels" not in ds.column_names:
        def add_labels(sample):
            sample["labels"] = copy.deepcopy(sample["input_ids"])
            return sample

        ds = ds.map(
            add_labels, 
            desc="Adding labels", 
            num_proc=args.num_proc,
            batched=True,
            batch_size=args.cpu_batch_size
        )
        changed_dataset = True
        logger.info(f"Added labels")
    else:
        logger.info(f"Dataset already has labels")

    # add attention mask
    if "attention_mask" not in ds.column_names:
        def add_attention_mask(sample):
            sample["attention_mask"] = torch.ones(
                len(sample["input_ids"]), dtype=torch.int8)
            return sample
        
        ds = ds.map(
            add_attention_mask, 
            desc="Adding attention mask", 
            num_proc=args.num_proc,
            batched=True,
            batch_size=args.cpu_batch_size
        )
        changed_dataset = True
        logger.info(f"Added attention mask")
    else:
        logger.info(f"Dataset already has attention mask")

    # truncate dataset
    if args.truncate:
        def truncate(sample):
            # todo, do for all columns
            sample["input_ids"] = sample["input_ids"][0:args.truncate]
            sample["labels"] = sample["labels"][0:args.truncate]
            sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
            return sample
        
        ds = ds.map(
            truncate, 
            desc="Truncating", 
            num_proc=args.num_proc, 
            batched=True, 
            batch_size=args.cpu_batch_size
        )
        changed_dataset = True
    else:
        logger.info(f"Dataset does not need truncation")

    # add num_tokens
    if "num_tokens" not in ds.column_names:
        ds = ds.map(lambda batch: {"num_tokens": [len(ids) for ids in batch["input_ids"]]}, batched=True, batch_size=args.cpu_batch_size, num_proc=args.num_proc)
        changed_dataset = True
        logger.info(f"Added num_tokens")
    else:
        logger.info(f"Dataset already has num_tokens")

    # save dataset
    if args.save_dataset_path and changed_dataset: 
        save_path = args.save_dataset_path
        logger.info(f"Saving dataset to {save_path}")
        ds.save_to_disk(save_path)


if __name__ == "__main__":
    try:
        threads = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
    except:
        threads = os.cpu_count()

    args = argparse.ArgumentParser()
    args.add_argument("--cpu-batch-size", type=int, default=1000)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--model", type=str,
                      default="meta-llama/Llama-3.2-1B")
    args.add_argument("--truncate", type=int)
    args.add_argument("--dataset-path", type=str, default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    args.add_argument("--raw-data-name", type=str, default="genqa")
    args.add_argument("--num-proc", type=int, default=threads)
    args.add_argument("--save-dataset-path", type=str)
    args.add_argument("--tokenizer-path", type=str)
    args.add_argument("--pretokenizer-name", type=str)
    args.add_argument("--force-tokenize", action="store_true")
    args.add_argument("--task", type=str, default="default")
    main(args.parse_args())
