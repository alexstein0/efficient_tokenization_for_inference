import logging

from efficient_tokenization.tokenize_simple import my_tokenize, get_genqa_data, get_tokenizer, get_magpie_data, apply_chat_template, create_translation_dataset_with_template, get_gsm8k_data, get_mbpp_data, create_dictionary_dataset
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import copy
import torch
import os
import psutil
import argparse
from typing import List
from chat_templating import visualize_loss_mask

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
    
def get_preprocessed_data(ds, args):
    if args.raw_data_name == "genqa":
        ds = get_genqa_data(ds, track_role=True, batch_size=args.cpu_batch_size, threads=args.num_proc)
    elif args.raw_data_name == "magpie":
        ds = get_magpie_data(ds, track_role=True, batch_size=args.cpu_batch_size, threads=args.num_proc)
    elif args.raw_data_name == "gsm8k":
        ds = get_gsm8k_data(ds, track_role=True, batch_size=args.cpu_batch_size, threads=args.num_proc)
    elif args.raw_data_name == "mbpp":
        ds = get_mbpp_data(ds, track_role=True, batch_size=args.cpu_batch_size, threads=args.num_proc)
    elif args.raw_data_name == "dictionary":
        print("Dictionary data")
    else:
        raise ValueError(f"Unsupported raw data name: {args.raw_data_name}")
    return ds
    
def preprocess_dataset(ds, args, tokenizer, logger, task_name):
    # tokenize dataset
    changed_dataset = False

    if args.force_tokenize or "input_ids" not in ds.column_names:
        logger.info(f"Tokenizing dataset")
        ds = my_tokenize(ds.select_columns("text"), tokenizer, args.cpu_batch_size, args.num_proc)
        changed_dataset = True
        logger.info(f"Tokenized dataset")
    else:
        logger.info(f"Dataset already tokenized")

    # TODO for all these checks make sure every row has the value populated

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

    if "task_type" not in ds.column_names:
        def add_task_type(batch):
            batch["task_type"] = [task_name] * len(batch["input_ids"])  # Assuming 'input_ids' is a column in your dataset
            return batch

        ds = ds.map(
            add_task_type, 
            desc="Adding task type", 
            num_proc=args.num_proc,
            batched=True,
            batch_size=args.cpu_batch_size
        )
        changed_dataset = True
        logger.info(f"Added task type")
    else:
        logger.info(f"Dataset already has task type")

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

    return ds, changed_dataset


def main(args):
    logger = setup_logging()
    # Get tokenizer
    logger.info(f"Getting tokenizer")

    base_tokenizer = None
    if args.tokenizer_path is not None:
        # load tokenizer from vocab file
        # try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        logger.info(f"Loaded HF tokenizer from vocab file: {args.tokenizer_path}")
        base_tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info(f"Loaded base tokenizer from model: {args.model}")
        # except:
        #     base_tokenizer = AutoTokenizer.from_pretrained(args.model)
        #     vocab_file_path = args.tokenizer_path
        #     pre_tok_name = args.pre_tok_name
        #     tokenizer = get_tokenizer(vocab_file_path, pre_tok_name=pre_tok_name, old_tokenizer=base_tokenizer)
        #     logger.info(f"Loaded tokenizer from .model file: {vocab_file_path} and pre_tok_name: {pre_tok_name}")
    else:
        # get original_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info(f"Loaded tokenizer from model: {args.model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token


    # Get Dataset
    # the data must be fully preprocessed and ready to be tokenized with the tokenized text in "text" column
    dataset_path = args.dataset_path

    logger.info(f"Downloading dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)

    if args.dry_run:
        try:
            ds = ds.select(range(100))
        except:
            ds = ds["train"].select(range(100))
        args.dictionary_total_samples = 100
    
    print(ds)
    ds = get_preprocessed_data(ds, args)

    print(f"tasks: {args.task}")
    # this block is the only thing that is dataset specific
    ds_dict = {}
    if "default" in args.task:
        ds_dict["default"] = apply_chat_template(ds, tokenizer, args.cpu_batch_size, args.num_proc, args.chat_template_name)
        print(visualize_loss_mask(ds_dict["default"]["input_ids"][0], tokenizer, ds_dict["default"]["loss_mask"][0]))

    if "translation" in args.task:
        if base_tokenizer is None:
            base_tokenizer = AutoTokenizer.from_pretrained(args.model)
        ds_dict["translation"] = create_translation_dataset_with_template(ds, base_tokenizer, tokenizer, args.cpu_batch_size, args.num_proc, args.chat_template_name)

        print(visualize_loss_mask(ds_dict["translation"]["input_ids"][0], tokenizer, ds_dict["translation"]["loss_mask"][0]))

    if "dictionary" in args.task and args.dict_ds_path is not None:
        dict_ds = load_from_disk(args.dict_ds_path)
        dict_ds = dict_ds.select(range(args.num_added_tokens))
        ds_dict["dictionary"] = create_dictionary_dataset(dict_ds, base_tokenizer, tokenizer, args.dictionary_total_samples, args.min_words_per_sample, args.max_words_per_sample, args.chat_template_name)

    changed_dataset = False

    for task_name in ds_dict:
        task_str = f"{args.chat_template_name}-"
        if task_name == "dictionary":
            task_str += f"{args.dictionary_total_samples}samples-"
        dataset_path = os.path.join(args.save_dataset_dir, f"{args.raw_data_name}-{task_str}{args.save_dataset_name}-{args.num_added_tokens}")
        if os.path.exists(dataset_path):
            logger.info(f"Dataset already exists, skipping: {dataset_path}")
            continue

        print(f"processing dataset for task: {task_name}")
        print(ds_dict[task_name])
        ds = ds_dict[task_name]
        ds, changed_dataset = preprocess_dataset(ds, args, tokenizer, logger, task_name)

        # save dataset
        if args.save_dataset_name and changed_dataset and not args.dry_run: 
            logger.info(f"Saving dataset to {dataset_path}")
            ds.save_to_disk(dataset_path)

        # final_ds_list.append(ds)
        print("done processing dataset")
        print(ds)

    # ds = concatenate_datasets(final_ds_list)
    # del final_ds_list
    # del ds_list
    # del mixed_ds_list

def test_magpie_preprocessing():
    """Test the preprocessing pipeline with magpie data"""
    class Args:
        def __init__(self):
            self.raw_data_name = "magpie"
            self.dataset_path = "datasets/magpie_pro_300k_filtered"
            self.save_dataset_name = "tokenized_1"
            self.task = "default,translation"
            self.model = "meta-llama/Llama-3.2-1B-Instruct"
            self.tokenizer_path = "/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1"
            self.cpu_batch_size = 1000
            # self.num_proc = 24
            self.save_dataset_dir = "datasets"
            self.force_tokenize = True
            self.pre_tok_name = None
            self.truncate = None

    args = Args()
    args.task = args.task.split(",")
    main(args)

if __name__ == "__main__":
    try:
        threads = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
    except:
        threads = os.cpu_count()

    args = argparse.ArgumentParser()
    args.add_argument("--cpu-batch-size", type=int, default=10)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--model", type=str,
                      default="meta-llama/Llama-3.2-1B")
    args.add_argument("--truncate", type=int)
    args.add_argument("--dataset-path", type=str)
    args.add_argument("--raw-data-name", type=str, default="genqa")
    args.add_argument("--num-proc", type=int, default=threads)
    args.add_argument("--save-dataset-name", type=str)
    args.add_argument("--save-dataset-dir", type=str, default="datasets")
    args.add_argument("--dict-ds-path", type=str, default=None)
    args.add_argument("--dictionary-total-samples", type=int, default=1000)
    args.add_argument("--num-added-tokens", type=int, default=1000)
    args.add_argument("--min-words-per-sample", type=int, default=5)
    args.add_argument("--max-words-per-sample", type=int, default=15)
    args.add_argument("--tokenizer-path", type=str)
    args.add_argument("--pretokenizer-name", type=str)
    args.add_argument("--chat-template-name", type=str, default="llama32", choices=["llama32", "phi", "qwen"])
    args.add_argument("--force-tokenize", action="store_true")
    args.add_argument("--task", type=str, default="default")
    args.add_argument("--dry-run", action="store_true")
    # args.add_argument("--task_list_split", type=str, default=None)
    args = args.parse_args()

    args.task = args.task.split(",")
    print(args)

    # if args.task_list_split is None:
    #     print(f"WARNING: task_list_split is None, using uniform split")
    #     args.task_list_split = [1.0/len(args.task) for _ in args.task]
    # else:
    #     args.task_list_split = [float(x) for x in args.task_list_split.split(",")]
    #     if len(args.task_list_split) != len(args.task):
    #         print(f"WARNING: task_list_split length {len(args.task_list_split)} does not match task length {len(args.task)}, using uniform split")
    #         args.task_list_split = [1.0/len(args.task) for _ in args.task]
    #     else:
    #         total = sum(args.task_list_split)
    #         args.task_list_split = [x/total for x in args.task_list_split]

    main(args)
