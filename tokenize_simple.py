from tokenizers import pre_tokenizers, Regex
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from typing import Dict, List, Optional
# import logging
import psutil
from datasets import load_dataset, load_from_disk
import datasets
datasets.disable_caching()

import os
import copy

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


# logging.basicConfig(level=logging.INFO)

# batch_size = 500

# try:
#     threads = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
# except:
#     threads = os.cpu_count()


def extract_genqa(example_list: List[List[Dict[str, str]]], track_role: bool = False) -> List[List[str]]:
    output = []
    for text in example_list:
        raw_messages = []
        for message in text:
            if message.get("content"):
                this_message = ""
                if track_role and message.get('role') is not None:
                    # if message.get('role') == 'user':
                    #     this_message += "Q: "
                    # elif message.get('role') == 'assistant':
                    #     this_message += "A: "
                    this_message += f'<|{message.get("role")}|> '
                content = message.get('content', '').strip()
                this_message += content
                raw_messages.append(this_message.strip())
        # raw_messages = [message.get('content', '').strip() for message in text if message.get('content')]
        output.append("".join(raw_messages))
    return output
                  


def flatten_genqa_conversations(example: Dict[str, List[List[Dict[str, str]]]],
                                tokenizer: AutoTokenizer = None,
                                track_role: bool = False
                                ) -> Dict[str, List]:
    """
    Extracts and concatenates user and assistant messages, encodes them into bytes.
    
    Args:
        example (Dict): A single example from the dataset containing a 'text' field.
    
    Returns:
        Dict: A new field 'conversation_bytes' with a list of byte sequences.
    """

    text_col = example.get('text', [])
    extracted_texts = extract_genqa(text_col, track_role)
    
    if tokenizer is not None:
        all_texts, lengths = tokenize_text_with_pretokenizer(extracted_texts, tokenizer)
        return {'text': all_texts, 'num_tokens': lengths}
    else:
        return {'text': extracted_texts}
        # lengths.extend([len(x) for x in example_text])
        # output['text'] = all_texts



def tokenize_text_with_pretokenizer(extracted_texts: List[List[str]],
                                   tokenizer: AutoTokenizer) -> Dict[str, List]:
    
    lengths = []
    all_texts = []
    for example_text in extracted_texts:
        # raw_messages = [message.get('content', '').strip() for message in text if message.get('content')]
        tokenized_messages = []
        for i, message in enumerate(example_text):
            pre_tokenized_message = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(message)
            tokenized_message = []
            message_token_count = 0
            for subword, (start, end) in pre_tokenized_message:
                subword = message[start:end]
                tokenized_subword = tokenizer.tokenize(subword)
                tokenized_message.append(tokenized_subword)
                message_token_count += len(tokenized_subword)
            lengths.append(message_token_count)

            tokenized_messages.append(tokenized_message)
        all_texts.extend(tokenized_messages)
    return all_texts, lengths


def load_pretokenizer(pre_tok_name: str):
    pre_tokenizer_map = {
            "llama3": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
            "empty": None,
            "conjunctions": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)""",
            "???": r"""[^\r\n\p{L}\p{N}]?\p{L}+""",
            "numbers": r"""\p{N}{1,3}""",
            "???": r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            "???": r"""\s*[\r\n]+""",
            "???": r"""\s+""",
        }
    pre_tok_list = []
    if pre_tok_name != "empty":
        regex_obj = Regex(pre_tokenizer_map[pre_tok_name])
        pre_tok_obj = pre_tokenizers.Split(regex_obj, behavior="isolated", invert=False)
        pre_tok_list.append(pre_tok_obj)

    pre_tok_list.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    pre_tok = pre_tokenizers.Sequence(pre_tok_list)
    return pre_tok

def load_tokenizer_from_file(vocab_file_path: str):
    tok = PreTrainedTokenizerFast(vocab_file=vocab_file_path)
    return tok

def my_tokenize(ds, tok, batch_size, threads):
    def tokenize_function(example):
        return tok(
            example["text"], 
            # padding="max_length",  # Ensures uniform length
            truncation=True,       # Prevents sequences from exceeding model limit
            padding=False,  # Ensures uniform length
            max_length=2048,       # Adjust based on Llama3 context size
        )

    tokenized_dataset = ds.map(
        # lambda x: tok(x['text'], padding="max_length", truncation=True),  # Pass the entire text field
        tokenize_function,  # Pass the entire text field
        batched=True,
        batch_size=batch_size,
        num_proc=threads,  # Use multiple processes for faster tokenization
        desc="Tokenizing dataset"
    )

    def add_labels(example):
        example["labels"] = example["input_ids"].copy()  # Shifted labels for training
        return example

    tokenized_dataset = tokenized_dataset.map(
        add_labels,  # Copy input_ids to labels
        batched=True,
        batch_size=batch_size,
        num_proc=threads,  # Use multiple processes for faster tokenization
        desc="copying to labels"
    )
    return tokenized_dataset

def apply_pretokenizer(tokenizer, pre_tok_name: str):
    pre_tok = load_pretokenizer(pre_tok_name)
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tok
    return tokenizer

def get_tokenized_data(tokenizer_path: str, dataset_path: str, pre_tok_name: Optional[str] = None):
    tok = load_tokenizer_from_file(tokenizer_path)
    if pre_tok_name is not None:
        tok = apply_pretokenizer(tok, pre_tok_name)
    # logging.info(f"Tokenizing using {tokenizer_path}")
    ds = load_from_disk(dataset_path)
    ds = get_genqa_data(ds)
    
    tokenized_dataset = my_tokenize(ds.select_columns("text"), tok)

    return tok, ds, tokenized_dataset

def get_tokenizer(tokenizer_path: str, pre_tok_name: str = None):
    tok = load_tokenizer_from_file(tokenizer_path)
    if pre_tok_name is not None:
        tok = apply_pretokenizer(tok, pre_tok_name)
    return tok

# for Gen qa data
def get_genqa_data(ds: datasets.Dataset, tokenizer = None, track_role: bool = False, threads=1, batch_size=1000):
    fn_kwargs={
        "tokenizer": tokenizer,
        "track_role": track_role,
        }

    # logging.info(f"First row of raw dataset: \n{ds.select_columns('text').select([0])['text']}")
    ds = ds.select_columns("text").map(
        flatten_genqa_conversations,
        num_proc=threads,
        batched=True,
        batch_size=batch_size,
        fn_kwargs=fn_kwargs,
        )
    # logging.info(f"First row of dataset after extraction: \n{ds.select_columns('text').select([0])['text']}")

    return ds

if __name__ == "__main__":

    raw_data_name = "genqa"
    pre_tok_name = "empty"
    ext = "math"


    tokenizer_path_old = f"/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-{ext}-{pre_tok_name}-start"
    tokenizer_file_old = "new_mergeable_ranks_2000.model"
    vocab_file_path = f"{tokenizer_path_old}/{tokenizer_file_old}"
    
    ds_path = f"/fs/cml-projects/llm-pretraining/datasets/raw/{raw_data_name}/{ext}"

    tok, ds = get_tokenized_data(vocab_file_path, ds_path, pre_tok_name=pre_tok_name)

