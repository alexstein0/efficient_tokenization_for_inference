import logging

from tokenize_simple import get_tokenized_data, flatten_genqa_conversations, my_tokenize, get_genqa_data, get_tokenizer
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

    # raw_data_name = "genqa" # TODO make configurable
    # ext = "math" # TODO make configurable
    # pre_tok_name = "empty" # TODO make configurable
    # tokenizer_path_old = f"/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-{ext}-{pre_tok_name}-start"
    # tokenizer_file_old = "new_mergeable_ranks_2000.model"
    #--tokenizer-path /cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start/new_mergeable_ranks_2000.model
    # vocab_file_path = f"{tokenizer_path_old}/{tokenizer_file_old}"
    # dataset_path = f"/fs/cml-projects/llm-pretraining/datasets/raw/{raw_data_name}/{ext}" # TODO make configurable
    # dataset_path = "/cmlscratch/astein0/efficient_tokenization_for_inference/datasets/test"
    # max_seq_length = 2048 # TODO make configurable and usable
    # save_path = os.path.join(args.save_dataset_path, f"{raw_data_name}-{ext}-{pre_tok_name}-start")

    # Get tokenizer
    logger.info(f"Getting tokenizer")

    if args.tokenizer_path is not None:
        # load tokenizer from vocab file
        vocab_file_path = args.tokenizer_path
        tokenizer = get_tokenizer(vocab_file_path, args.pretokenizer_name)
        logger.info(f"Loaded tokenizer from vocab file: {vocab_file_path}")
    else:
        # get original_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Loaded tokenizer from model: {args.model}")

    # Get Dataset
    # the data must be fully preprocessed and ready to be tokenized with the tokenized text in "text" column
    dataset_path = args.dataset_path

    logger.info(f"Downloading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)
    changed_dataset = False

    # this is the only thing that is dataset specific
    ds = preprocess_data(ds, args)

    # tokenize dataset
    if args.force_tokenize or "input_ids" not in ds.column_names:
        logger.info(f"Tokenizing dataset")
        ds = my_tokenize(ds.select_columns("text"), tokenizer, args.cpu_batch_size, args.num_proc)
        changed_dataset = True
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
    main(args.parse_args())
