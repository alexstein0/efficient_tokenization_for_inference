from datasets import Dataset, load_from_disk, load_dataset, DatasetDict
import psutil
import os
import regex
import collections
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Any, Dict, Tuple, Optional, Generator
import logging
import base64
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import re
import matplotlib.pyplot as plt
import time

from typing import Dict, Any

import json

import datasets

datasets.disable_caching()

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

# from tokenizers import pre_tokenizers, Regex

import argparse

from efficient_tokenization.tokenize_simple import load_pretokenizer, get_genqa_data, get_magpie_data, get_gsm8k_cot_data
from efficient_tokenization.tokenization_utils import SaveModule, read_training_info, read_tokenizer_from_model

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
# https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json
# added_token_list = ["```ĊĊ"]

def convert_data_to_text_col(my_dataset, fields: List[str], threads: int=24):
    def concat_columns(example: Dict[str, Any], cols: List[str]):
        text = ""
        for k in cols:
            text += example[k] + " "
        example["text"] = text[:-1]
        return example

    if "text" not in my_dataset.features:
        if len(fields) > 0:
            my_dataset = my_dataset.select_columns(fields)

        my_dataset = my_dataset.map(concat_columns, num_proc=threads, fn_kwargs={"cols": fields})  #, batched=True)

    my_dataset = my_dataset.select_columns("text")

    return my_dataset


def extract_conversations_batched(example: Dict[str, List[List[Dict[str, str]]]],
                                   tokenizer: AutoTokenizer = None) -> Dict[str, List]:
    """
    Extracts and concatenates user and assistant messages, encodes them into bytes.
    
    Args:
        example (Dict): A single example from the dataset containing a 'text' field.
    
    Returns:
        Dict: A new field 'conversation_bytes' with a list of byte sequences.
    """
    text_col = example.get('text', [])
    all_texts = []
    lengths = []
    for text in text_col:
        raw_messages = [message.get('content', '').strip() for message in text if message.get('content')]
        tokenized_messages = []
        if tokenizer is not None:
            for i, message in enumerate(raw_messages):
                pre_tokenized_message = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(message)
                tokenized_message = []
                message_token_count = 0
                for subword, (start, end) in pre_tokenized_message:
                    subword = message[start:end]
                    tokenized_subword = tokenizer.tokenize(subword)
                    tokenized_message.append(tokenized_subword)
                    message_token_count += len(tokenized_subword)
                lengths.append(message_token_count)

                # tokenized_message = tokenizer.tokenize(message)
                # lengths.append(len(tokenized_message))
                tokenized_messages.append(tokenized_message)
            all_texts.extend(tokenized_messages)
        else:
            all_texts.extend(raw_messages)
    
    return {'text': all_texts, "num_tokens": lengths}

# code = processed_dataset['code'].select_columns(['text'])
# math = processed_dataset['math'].select_columns(['text'])
# academic = processed_dataset['academic'].select_columns(['text'])

# import copy
# from transformers import PreTrainedTokenizerFast

# tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3")

# text = " decentralized"

# # 1) Manually run the existing pretokenizer
# pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
# if pre_tokenizer is None:
#     raise ValueError("No pre-tokenizer is configured for this tokenizer.")

# pretokenized = pre_tokenizer.pre_tokenize_str(text)
# # pretokenized might look like: [('Ġdecentralized', (0, 14))]

# # 2) Make a temporary copy of the tokenizer with the pre-tokenizer & normalizer disabled
# temp_tokenizer = copy.deepcopy(tokenizer)
# temp_tokenizer.backend_tokenizer.pre_tokenizer = None
# temp_tokenizer.backend_tokenizer.normalizer = None

# # 3) Now tokenize each pretokenized chunk without re-doing pre-tokenization
# grouped_tokens = []
# for subword, _ in pretokenized:
#     tokens = temp_tokenizer.tokenize(subword)
#     grouped_tokens.append(tokens)

# print("Grouped Tokens:", grouped_tokens)


# def my_tokenize(ds, tok):
#     tokenized_dataset = ds.map(
#         lambda x: tok(x['text']),  # Pass the entire text field
#         batched=True,
#         batch_size=batch_size,
#         num_proc=threads,  # Use multiple processes for faster tokenization
#         desc="Tokenizing dataset"
#     )
#     return tokenized_dataset



def bpe_encode(
    mergeable_ranks: dict[bytes, int], input_data: bytes
) -> list[int]:
    # Takes in a byte string and splits it into a byte list before running bpe to create tokens
    # Returns a list of those tokens
    parts = [bytes([b]) for b in input_data]
    parts = bpe(mergeable_ranks, parts, visualise=None)
    tokens = [mergeable_ranks[part] for part in parts]
    return tokens

def bpe(
    mergeable_ranks: dict[bytes, int], parts: List[bytes]
) -> list[bytes]:
    # Takes in a list of bytes and merges into tokens.
    # Returns the merged parts
    while True:
        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    return parts

def tokenize_sample_str(sample: str, mergeable_ranks: Dict, regex_str: str) -> List[List[bytes]]:
    # Use the regex to split the text into (approximately) words
    words = regex.findall(regex_str, sample)
    tokens = []
    for word in words:
        # Turn each word into tokens, using the byte pair encoding algorithm
        word_bytes = word.encode("utf-8")
        # word_tokens = bpe_encode(mergeable_ranks, word_bytes, visualise=None)
        parts = [bytes([b]) for b in word_bytes]
        word_tokens = bpe(mergeable_ranks, parts)
        tokens.append(word_tokens) 
        # TODO need to decide if this has the topkens split or not.  Currently joins the words together
    return tokens

# def encode_sample_bytes(parts: List[bytes], mergeable_ranks: Dict) -> List[List[int]]:
#     # Takes in the list of byte parts and bpes them.  
#     # Effectively skips the step where you split into bytes and merge for parts already known
#     parts = bpe(mergeable_ranks, parts, visualise=None)
#     tokens = [[mergeable_ranks[part] for part in parts]]
#     return tokens

def tokenize_batch(batch: List[str] | List[List[bytes]], mergeable_ranks: Dict, regex_str: str) -> List[List[List[bytes]]]:
    output = []
    for sample in batch:
        if isinstance(sample, str):
            tokens = tokenize_sample_str(sample, mergeable_ranks, regex_str)
        else:
            # list of bytes
            tokens = bpe(mergeable_ranks, sample)
        output.append(tokens)
    return output

def single_threaded_tokenize(data: Generator[str, None, None] | List[List[bytes]], mergeable_ranks: Dict, regex_str: str):
    tokenized_data = []
    for batch in tqdm(data, desc="Processing samples"):
        batch_tokens = tokenize_batch(batch, mergeable_ranks, regex_str)
        # flattened_batch = [token for sublist in batch_tokens for token in sublist]
        tokenized_data.append(batch_tokens)

    return tokenized_data


def distributed_tokenize(data, mergeable_ranks: Dict, regex_str: str) -> List[List[List[List[bytes]]]]:
    # output is batches x samples_per_batch x words_per_sample x bytes_per_word
    tokenized_data = []
    # Using ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_sample = {executor.submit(tokenize_batch, batch, mergeable_ranks, regex_str): batch for batch in data}
        
        # Create a tqdm progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(future_to_sample), desc="Processing samples"):
            batch_tokens = future.result()
            # flattened_batch = [token for sublist in batch_tokens for token in sublist]
            tokenized_data.append(batch_tokens)

    return tokenized_data




# ------------------------------------------------------------------------
# 1) Pair Counting in Parallel
# ------------------------------------------------------------------------
def count_pairs_parallel(shard: Dataset) -> Tuple[collections.Counter, int]:
    """
    Count the frequency of each consecutive byte pair in the given chunk.

    Args:
        chunk (List[List[List[bytes]]]): A subset of samples to process, where each sample is
                                         a list of pieces, and each piece is a list of bytes.

    Returns:
        Tuple[collections.Counter, int]: A counter of byte pairs and the number of tokens processed.
    """
    # logger.info(f"Process {os.getpid()} is handling a chunk of size {len(chunk)}")
    current_stats = collections.Counter()
    current_token_count = 0
    
    for sample in shard['text']:              # sample is a list of pieces
        for piece in sample:          # piece is a list of bytes
            current_token_count += len(piece)
            for pair in zip(piece[:-1], piece[1:]):
                current_stats[pair] += 1

    return current_stats, current_token_count


def reduce_counters(counters_and_counts):
    total_stats = collections.Counter()
    total_tokens = 0
    for (ctr, cnt) in counters_and_counts:
        total_stats.update(ctr)
        total_tokens += cnt
    return total_stats, total_tokens

def find_common_pair_parallel(dataset: Dataset) -> Tuple[Tuple[bytes, bytes], int, int]:
    """
    Splits the dataset into shards for each worker using dataset.shard(...).
    Each worker counts pairs in parallel, and we reduce them to find the single most common pair.
    Returns (most_common_pair, occurrences, total_token_count).
    """
    num_workers = min(len(dataset), concurrent.futures.ProcessPoolExecutor()._max_workers)

    # Spawn parallel processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            shard = dataset.shard(num_shards=num_workers, index=i, contiguous=True)
            futures.append(executor.submit(count_pairs_parallel, shard))

        counters_and_counts = [f.result() for f in futures]

    total_stats, total_token_count = reduce_counters(counters_and_counts)
    if len(total_stats) == 0:
        logger.info("No more candidate merges to make")
        return (None, -1, total_token_count)

    most_common_pair = max(total_stats, key=total_stats.get)
    return (most_common_pair, total_stats[most_common_pair], total_token_count)

# import uuid
# import shutil
def do_merge_iteration(
    dataset: Dataset,
    iteration_id: int,
    most_common_pair: Tuple[bytes, bytes],
    cache_file_name: Optional[str],
    batch_size: int,
    threads: int
) -> Dataset:
    """
    Perform one iteration of merges:
      1) Create a temp directory for arrow shards
      2) Map over dataset -> partial shards + final arrow file in that directory
      3) Wait & remove the directory to avoid leftover shards
    """
    # Create a unique temp directory for this iteration
    temp_folder = os.environ["HF_DATASETS_CACHE"]
    os.makedirs(temp_folder, exist_ok=True)
    cache_file_name = f"{temp_folder}/{cache_file_name}.arrow"

    dataset = dataset.map(
        merge_data_in_parallel,
        batched=True,
        batch_size=batch_size,
        num_proc=threads,
        desc=f"Merges iteration {iteration_id}",
        load_from_cache_file=False,  # Needs to be false to get get updated version
        fn_kwargs={"most_common_pair": most_common_pair},
        # The key lines:
        cache_file_name=cache_file_name,      # this is needed because it allows the cache to overwrite itself in a location that you specify.  Otherwise the files get unruly
        keep_in_memory=False,         # we want an on-disk arrow
    )

    # logger.info(dataset.cleanup_cache_files())

    return dataset

# ------------------------------------------------------------------------
# 2) Merging Data in Parallel (the .map function)
# ------------------------------------------------------------------------
def merge_data_in_parallel(batch: Dataset, most_common_pair: Tuple[bytes, bytes]) -> Dict[str, List[List[bytes]]]:
    """
    Merges the most_common_pair in each row of 'text'.
    'batch' is a dictionary: batch["text"] is a list of rows, each row is a list of lists of bytes.
    """
    token_bytes = most_common_pair[0] + most_common_pair[1]
    updated_samples = []

    for sample in batch["text"]:
        new_sample = []
        for piece in sample:
            new_piece = []
            i = 0
            while i < len(piece) - 1:
                if (piece[i], piece[i + 1]) == most_common_pair:
                    new_piece.append(token_bytes)
                    i += 2
                else:
                    new_piece.append(piece[i])
                    i += 1
            if i == len(piece) - 1:
                new_piece.append(piece[i])
            new_sample.append(new_piece)
        updated_samples.append(new_sample)

    return {"text": updated_samples}

# ------------------------------------------------------------------------
# 3) Single Loop Over the Dataset - Merges Then Updates
# ------------------------------------------------------------------------
def single_loop_parallel(dataset: Dataset, 
                         ranks: Dict[bytes, int], 
                        #  original_size: int = -1, 
                         cache_file_name: Optional[str] = None,
                         batch_size: int = 1000,
                         threads: int = 16
                         ) -> Tuple[Dataset, Dict[bytes, int], Tuple[bytes, bytes], int]:
    """
    1) Finds the most common pair in 'dataset' using parallel shards.
    2) Creates a new token in 'ranks'.
    3) Calls dataset.map(...) to merge that pair, writing to a temporary arrow file.
    4) Removes partial shard files.

    Returns the new dataset, updated ranks, the most_common_pair, and total token count.
    """
    hardcode_count_pairs = False
    # start_par = time.perf_counter()
    # 1) Find the most common pair in parallel
    if hardcode_count_pairs:
        most_common_pair_list = [
            ["``", "`ĊĊ"],
            [",", "Ġ"],
            ["Ġthe", "Ġ`"],
            ["Ċ", "ĠĠĠĠ"],
            ["Ġof", "Ġthe"],
        ]
        most_common_pair = most_common_pair_list[len(ranks) - 128000]
        occurrences = -1
        current_token_count = -1
    else:
        most_common_pair, occurrences, current_token_count = find_common_pair_parallel(dataset)
    if most_common_pair is None:
        return None, None, None
    
    # 2) Add new token to ranks
    # new_merges.append(most_common_pair)
    token_bytes = most_common_pair[0] + most_common_pair[1]
    if token_bytes in ranks:
        logger.info(f"CANDIDATE {token_bytes} ALREADY IN RANKS! found with {ranks[token_bytes]}")
        # return
    new_token_id = len(ranks)
    ranks[token_bytes] = new_token_id
    dataset = do_merge_iteration(dataset, new_token_id, most_common_pair, cache_file_name, batch_size, threads)
    # end_par = time.perf_counter()
    # par_time3 = end_par - start_par
    # new_token_count = current_token_count - occurrences
    
    # compression_rate = 1
    # if original_size > 0:
    #     compression_rate = new_token_count / original_size
    
    # logger.info(
    #     f"Adding token: {new_token_id:6d}  "                                                   # Left-align token with padding
    #     f"Bytes: {token_bytes:20s}  "                                                   # Display token bytes
    #     f"Pair: {most_common_pair[0]:10s} {most_common_pair[1]:10s}  "                  # Display merge bytes
    #     f"Pair ids: {ranks[most_common_pair[0]]:6d} {ranks[most_common_pair[1]]:6d}  "  # Display merge bytes
    #     f"Occurrences: {occurrences:10d}  "                                             # Left-align occurrences with padding
    #     f"Dataset size: {new_token_count:10d} tokens  "
    #     f"Compression Rate: {compression_rate:4.3f} compression  "
    #     f"Loop time: {par_time3}"
    # )
    return dataset, ranks, most_common_pair, occurrences #current_token_count



def single_loop(dataset: List[List[bytes]], ranks: Dict[bytes, int], original_size: int = -1) -> Tuple[List[List[List[bytes]]], int, int]:
    # Time the sequential version
    start_seq = time.perf_counter()
    most_common_pair, occurrences, current_token_count = find_common_pair(dataset)
    end_seq = time.perf_counter()
    seq_time = end_seq - start_seq

    # # Print timing information and results
    logger.info(f"Sequential version took: {seq_time:.6f} seconds")

    if most_common_pair is None:
        return None, None, None
    
    # new_merges.append(most_common_pair)
    token_bytes = most_common_pair[0] + most_common_pair[1]
    token = len(ranks)
    # Add the new token!
    if token_bytes in ranks:
        logger.info(f"CANDIDATE {token_bytes} ALREADY IN RANKS! found with {ranks[token_bytes]}")
        # return
    
    new_token_id = len(ranks)
    ranks[token_bytes] = token
    # Get the current timestamp
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # # Print the formatted string with enhanced formatting
    # logger.info(f"[{timestamp:20s}]  "
    #     f"Adding token: {token:6d}  "  # Left-align token with padding
    #     f"Bytes: {token_bytes:20s}  "      # Display token bytes
    #     f"Pair: {most_common_pair[0]:10s} {most_common_pair[1]:10s}  "      # Display merge bytes
    #     f"Pair ids: {ranks[most_common_pair[0]]:6d} {ranks[most_common_pair[1]]:6d}  "      # Display merge bytes
    #     f"Occurrences: {occurrences:10d}  "  # Left-align occurrences with padding
    #     f"Dataset size: {current_token_count:10d} tokens")

    dataset = merge_data(dataset, most_common_pair)

    end_seq2 = time.perf_counter()
    seq_time3 = end_seq2 - start_seq
    new_token_count = current_token_count - occurrences
    compression_rate = 1
    if original_size > 0:
        compression_rate = new_token_count / original_size
    logger.info(
        f"Adding token: {new_token_id:6d}  "                                                   # Left-align token with padding
        f"Bytes: {token_bytes:20s}  "                                                   # Display token bytes
        f"Pair: {most_common_pair[0]:10s} {most_common_pair[1]:10s}  "                  # Display merge bytes
        f"Pair ids: {ranks[most_common_pair[0]]:6d} {ranks[most_common_pair[1]]:6d}  "  # Display merge bytes
        f"Occurrences: {occurrences:10d}  "                                             # Left-align occurrences with padding
        f"Dataset size: {new_token_count:10d} tokens  "
        f"Compression Rate: {compression_rate:4.3f} compression  "
        f"Loop time: {seq_time3:5.3f}"
    )
    return dataset, ranks, most_common_pair, current_token_count

# Find the most common pair. This will become our next token
def find_common_pair(samples_of_words: List[List[List[bytes]]]) -> Tuple[bytes, int, int]:
    current_token_count = 0
    stats = collections.Counter()
    for sample in samples_of_words:
        for piece in sample:
            current_token_count += len(piece)
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1

    if len(stats) == 0:
        logger.info(f"No more candidate merges to make")
        return None, -1, current_token_count
    most_common_pair = max(stats, key=lambda x: stats[x])
    return most_common_pair, stats[most_common_pair], current_token_count

#
def merge_data(samples_of_words: List[List[List[bytes]]], most_common_pair: bytes) -> List[List[List[bytes]]]:
    # Now merge that most common pair in all the words. That is, update our training data
    # to reflect our decision to make that pair into a new token.
    updated_samples = []
    token_bytes = most_common_pair[0] + most_common_pair[1]
    for sample_num, sample in enumerate(samples_of_words):
        new_sample = []
        for piece in sample:
            new_piece = []
            i = 0
            while i < len(piece) - 1:
                if (piece[i], piece[i + 1]) == most_common_pair:
                    new_piece.append(token_bytes)
                    i += 2
                else:
                    new_piece.append(piece[i])
                    i += 1
            if i == len(piece) - 1:
                new_piece.append(piece[i])
            new_sample.append(new_piece)
        updated_samples.append(new_sample)
    return updated_samples


def pretokenize_data_before_training(data: List[str] | Generator[str, None, None] | Generator[List[str], None, None], 
                                    #  new_pat_str: str = None, 
                                     old_pat_str: str = None, 
                                     initial_ranks: dict[bytes, int] = None,
                                     ) -> List[List[bytes]]:
       
    samples_of_words = distributed_tokenize(data, initial_ranks, old_pat_str)
    # samples_of_words = single_threaded_tokenize(data, initial_ranks, old_pat_str)
    # At this point, we have a list of samples, where each sample contains a list of words, where each word is a list of bytes. 
    # Merges can occur across words within a sample (depending on split rules), but not across samples

    # samples_of_words = [byte for sample in samples_of_words for sublist in sample for byte in sublist]
    samples_of_words = [[[byte for sublist in sample for byte in sublist] for sample in batch ] for batch in samples_of_words]  # this flattens the samples by removing spaces between "words"
                
    def list_to_generator(data_list):
        for item in data_list:
            yield item
            
    samples_of_words = list_to_generator(samples_of_words)
    samples_of_words = distributed_tokenize(samples_of_words, initial_ranks, None)
    # samples_of_words = single_threaded_tokenize(samples_of_words, initial_ranks, None)

    # flatten
    samples_of_words = [sample for batch in samples_of_words for sample in batch]
    return samples_of_words


def compare_tokenizations(ds_raw: Dataset, tokenizer_path_old: str, tokenizer_file_old: str, pre_tok):
    # ds_raw = load_from_disk(f"/fs/cml-projects/llm-pretraining/datasets/raw/{raw_data_name}/{ext}")
    first = ds_raw[0]["text"]
    first = [text.get('content', '').strip() for text in first if text.get('content')]
    first = first[1]
    print(first)
    # first = " ".join(first)
    tok1 = PreTrainedTokenizerFast(vocab_file="/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer/tokenizer.model")
    tok2 = PreTrainedTokenizerFast(vocab_file=f"{tokenizer_path_old}/{tokenizer_file_old}")
    tok2.backend_tokenizer.pre_tokenizer = pre_tok
    old_tokenization = tok1.tokenize(first)
    new_tokenization = tok2.tokenize(first)
    print(f"{len(old_tokenization)} {old_tokenization}")
    print(f"{len(new_tokenization)} {new_tokenization}")
    i = 0
    j = 0
    comp = 0
    while i < len(old_tokenization):
        if old_tokenization[i] != new_tokenization[j]:
            # only works if new tokenizer is combo of old
            new_token = new_tokenization[j]
            old_token = old_tokenization[i]
            print(f"{i}: {old_token}, {j}: {new_token}")
            loc = 0
            while loc + len(old_token) < len(new_token):
                print(f"{i}: {old_token}")
                loc += len(old_token)
                i += 1
                comp += 1
                old_token = old_tokenization[i]
        i += 1
        j += 1
    print(comp)

# Adapted from tiktokenizer code above
def bpe_continue_train(
    data: Dataset, 
    vocab_size: int, 
    # new_pat_str: Optional[str] = None, 
    old_pat_str: Optional[str] = None, 
    initial_ranks: Optional[Dict[bytes, int]] = None,
    is_pretokenized=False,
    parallel: bool=True,
    old_data: Dict = {},
    batch_size: int = 1000,
    threads: int = 16,
    save_interval: List[int] =[],
    cache_file_name: Optional[str] = None,
    # save_loc: str = "",
    # save_file_name: str = "",
    # base_tokenizer: str | PreTrainedTokenizerFast = None
    save_module: SaveModule = None,
) -> Tuple[Dict[bytes, int], Dict[bytes, int], List[int]]:
    logger.info(f"Initializing with mergeable ranks of size {len(initial_ranks)} and adding {vocab_size - len(initial_ranks)} new tokens")
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    
    if vocab_size < len(initial_ranks):
        raise ValueError("vocab_size must be greater than the number of initial ranks")
    
    if not is_pretokenized:
        data = pretokenize_data_before_training(data, old_pat_str, initial_ranks)
        # TODO assert that it is a list of list of lists

    if not parallel:
        flattened_data = []
        for batch in data:
            flattened_data.extend(batch)
        data = flattened_data
        del flattened_data

    # TODO If using a new pattern, we need to group the words in a way that will represent the new split not the old one.  
    
    # Now, use our data to figure out which merges we should make
    logger.info("Preprocess complete, starting merges")
    new_merges = old_data.get("merges", [])
    dataset_sizes = old_data.get("sizes", [])
    # these two must be present
    current_dataset_size = old_data.get("current_dataset_size")
    original_dataset_size = old_data.get("initial_size")
    
    step = 1
    while len(initial_ranks) < vocab_size:
        #  Need to make sure that data is samples x pieces x tokens x bytes
        start_timer = time.perf_counter()
        if parallel:
            data, initial_ranks, most_common_pair, occurrences = single_loop_parallel(data, initial_ranks, cache_file_name, batch_size, threads)
        else:
            raise NotImplementedError("Not implemented with the correct occurrences")
            # data, initial_ranks, most_common_pair, dataset_length = single_loop(data, initial_ranks, original_dataset_size)  # TODO this hasnt been updated to use occurrences

        end_timer = time.perf_counter()
        duration = end_timer - start_timer
        current_dataset_size -= occurrences
        compression_rate = current_dataset_size / original_dataset_size

        logger.info(
            f"Adding token: {len(initial_ranks) - 1:6d}  "  # Left-align token with padding
            f"Bytes: {most_common_pair[0] + most_common_pair[1]:20s}  "  # Display token bytes
            f"Pair: {most_common_pair[0]:10s} {most_common_pair[1]:10s}  "  # Display merge bytes
            f"Pair ids: {initial_ranks[most_common_pair[0]]:6d} {initial_ranks[most_common_pair[1]]:6d}  "  # Display merge bytes
            f"Occurrences: {occurrences:10d}  "  # Left-align occurrences with padding
            f"Dataset size: {current_dataset_size:10d} tokens  "
            f"Compression Rate: {compression_rate:4.3f} compression  "
            f"Loop time: {duration}"
        )

        new_merges.append(most_common_pair)
        dataset_sizes.append(current_dataset_size)
        # if (step % save_interval) == 0 and save_module is not None:
        if step in save_interval and save_module is not None:
            additional_info = {
                "state": step,
            }
            print(f"Saving at step {step}")
            save_module.save(merges=new_merges, dataset_sizes=dataset_sizes, additional_info=additional_info, ranks=initial_ranks, save_ext="-step")
            # save_path = f"{save_loc}/{save_file_name}"
            # new_added_tokens = get_new_added_tokens(initial_ranks, new_merges)
            # added_tokens = vocab_size - len(initial_ranks)
            
            # training_info = {
            #                 "state": step,
            #                 "merges": new_merges,
            #                 "new_tokens": new_added_tokens,
            #                 "number_new_tokens": added_tokens,
            #                 "initial_size": original_dataset_size,
            #                 "sizes": dataset_sizes,
            #                 "save_path": save_path,
            #                 # "base_tokenizer_path": ,
            #                 # "duration": "0",
            #                 # "pretokenizer": args.pre_tok_name,
            #                 # "cont_or_start": args.cont_or_start,
            #                 # "save_tokenized_data": args.save_tokenized_data,
            #                 # "old_tokenzer_info": old_tokenizer_info
            #             }
            # if base_tokenizer is not None:
            #     logger.info(f"Saving tokenizer.json (huggingface) file at step {step} to {save_path}")
            #     base_tokenizer_path = base_tokenizer if isinstance(base_tokenizer, str) else base_tokenizer.name_or_path
            #     pretokenizer = base_tokenizer.backend_tokenizer.pre_tokenizer
            #     training_info["pretokenizer"] = pretokenizer
            #     training_info["base_tokenizer_path"] = base_tokenizer_path
            #                         "pretokenizer_string": pre_tok.pre_tokenize_str,

            #     save_tokenizer_new(training_info, save_loc, base_tokenizer_path)
            # else:
            #     logger.info(f"Saving tokenizer.model file at step {step} to {save_path}")
            #     save_tokenizer(training_info, save_loc, save_file_name)

            # save_training_info(training_info, save_loc, save_file_name)
        step += 1

    return initial_ranks, new_merges, dataset_sizes



def get_data_generator(ds: Dataset, batch_size: int) -> Tuple[Generator[List[str], None, None], int]:
    def get_number_of_batches(dataset_length: int, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        return (dataset_length + batch_size - 1) // batch_size  # Equivalent to math.ceil(dataset_length / batch_size)

    def batch_iterator() -> Generator[List[str], None, None]:
        for i in range(0, len(ds), batch_size):
            # Use the Hugging Face dataset's select method to get the batch
            rows = ds[i: i + batch_size]
            yield rows["text"]  # Assuming "text" is the column name

    return batch_iterator(), get_number_of_batches(len(ds), batch_size)

def first_batch_generator(ds: Dataset, batch_size: int) -> Generator[List[str], None, None]:
    # Get the main batch iterator
    main_iterator, _ = get_data_generator(ds, batch_size)
    
    # Yield only the first batch
    yield next(main_iterator)  # Call next on the iterator


def get_data(cont_or_start: str, force_retokenize: bool, save_tokenized_data: bool, 
             dataset_path: str, tokenized_data_name: str, ext: str, 
            #  tokenizer_path_old: str, tokenizer_file_old: str, 
             tokenizer, base_tokenizer_path,
             raw_data_name: str, pre_tok, 
             threads: int, batch_size: int):
    logger.info("GETTING DATA")
    try:
        if cont_or_start == "cont":
            logger.info("Continuing tokenizer training so retokenizing and not saving")
            save_tokenized_data = False
            raise Exception("Continuing tokenizer training so retokenizing and not saving")
        if force_retokenize:
            logger.info("Forcing retokenization of data")
            raise Exception("Forcing retokenization of data")
        # ds = load_from_disk(f"/fs/cml-projects/llm-pretraining/datasets/processed/{tokenized_data_name}/{ext}") #.select_columns('text')
        ds = load_from_disk(os.path.join(dataset_path, tokenized_data_name)) #.select_columns('text')
        logger.info("Dataset loaded")
    except:
        logger.info(f"Downloading and processing raw dataset: {raw_data_name}")
        tok = tokenizer
        # tok = PreTrainedTokenizerFast(vocab_file=os.path.join(tokenizer_path_old, tokenizer_file_old))
        tok.backend_tokenizer.pre_tokenizer = pre_tok
        # logger.info(f"Tokenizing using {tokenizer_path_old}/{tokenizer_file_old}")
        # ds = load_from_disk(f"/fs/cml-projects/llm-pretraining/datasets/raw/{raw_data_name}/{ext}")
        ds = load_from_disk(os.path.join(dataset_path, raw_data_name))
        # ds = ds.select_columns("text").map(
        #     extract_conversations_batched,
        #     num_proc=threads,
        #     batched=True,
        #     batch_size=batch_size,
        #     fn_kwargs={"tokenizer": tok}
        # )
        if raw_data_name == "genqa":
            ds = get_genqa_data(ds, tokenizer=tok, track_role=False, threads=threads, batch_size=batch_size)
        elif raw_data_name == "magpie_pro_300k_filtered":
            ds = get_magpie_data(ds, tokenizer=tok, track_role=False, threads=threads, batch_size=batch_size)
        elif raw_data_name.startswith("gsm8k"):
            ds = get_gsm8k_cot_data(ds, tokenizer=tok, track_role=False, threads=threads, batch_size=batch_size)
        else:
            pass

        if save_tokenized_data:
            print(f"Saving tokenized data to {dataset_path}/{tokenized_data_name}")
            save_data_dir = os.path.join(dataset_path, tokenized_data_name)
            os.makedirs(save_data_dir, exist_ok=True)
            ds.save_to_disk(save_data_dir)

    logger.info(ds)
    return ds

def sort_vocab_and_print_new(mergeable_ranks: Dict[bytes, int]):
    sorted_vocab = {k: v for k, v in sorted(mergeable_ranks.items(), key=lambda item: item[1])}
    for k, v in sorted_vocab.items():
        if v >= 128000:
            print(k, v)


#### PLOTTING CODE
def extract_dataset_tokens(file_path):
    dataset_tokens = []
    # Regular expression to match the desired line format
    pattern = r"Dataset size:\s+(\d+)\s+tokens"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the <Z> value (the number of tokens)
                dataset_tokens.append(int(match.group(1)))

    return dataset_tokens


def plot_dataset_tokens(tokens_list_dict):
    plt.figure(figsize=(12, 6))  # Set the figure size
    # plt.plot(tokens_list.keys(), tokens_list.values(), marker='o', linestyle='-')  # Line plot with markers
    # Assign colors and markers
    # Assign 3 colors and 4 markers
    colors = ["red", "blue", "green"]  # Define your 3 colors
    markers = ["o", "^", "s", "D", ]  # Define your 4 markers
    linestyles = ["-", "--", "o", ".", ]  # Define your 4 linestyles
    linestyle = 0
    marker = 0
    color = 0
    freq = 50
    for linestyle, (pretok, group_dict) in enumerate(tokens_list_dict.items()):
        for color, (name, values) in enumerate(group_dict.items()):
            print(marker, color, pretok, name)
            plotted_values = [x for i, x in enumerate(values) if i % freq == 0]
            x_vals = [x*freq for x in range(len(plotted_values))]
            # plt.plot(range(len(values)), values, marker='o', linestyle='-', label=name, markersize=3)  # Line plot with markers
            plt.plot(x_vals, plotted_values, color=colors[color], marker=markers[marker], linestyle=linestyles[linestyle], label=f"{name}", markersize=3)  # Line plot with markers
    plt.title('Compression Rate')  # Title of the plot
    plt.xlabel('tokens Added')  # X-axis label
    plt.ylabel('Compression Rate')  # Y-axis label
    plt.grid(True)  # Show grid
        # Set x-ticks at intervals of 5 (or any other number)
    plt.xticks(ticks=range(0, 2000, 50), rotation=45)  # Rotate labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.legend()
    plt.show()  # Display the plot

def extract_wrapper(file_names: Dict[str, str]) -> Dict[str, List[int]]:
    output = {}
    folder_path = f'/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/log/'
    for pretok_name, group_dict in file_names.items():
        output_group_dict = {}
        for name, ext in group_dict.items():
            file_path = os.path.join(folder_path, ext)
            tokens_list = extract_dataset_tokens(file_path)
            if len(tokens_list) == 0:
                continue

            starting_count = tokens_list[0]
            tokens_list = [1-(x / starting_count) for x in tokens_list]
            output_group_dict[name] = tokens_list
        output[pretok_name] = output_group_dict
    return output



def plot_compression_rate():
    # file_exts = {
    #     "code": "tokenizer_-GLoPQ0_1.log",
    #     # "code2": "tokenizer_pjab0eI_1.log",
    #     "math": "tokenizer_JnoDfCA_1.log",
    #     "academic": "tokenizer_hk06tes_1.log"
    # }
    file_exts = {
        "empty": {
            "code": "tokenizercodeempty_W9OmRFY_1.log",
            "math": "tokenizermathempty_t4DO-4E_1.log",
            "acadmic": "tokenizeracademicempty_yCSY3uQ_1.log",
        },
        "llama": {
            "code":"tokenizercodellama3_IWh8aQY_1.log",
            "math":"tokenizermathllama3_F5dN4hA_1.log",
            "academic": "tokenizeracademicllama3_-TFtthk_1.log"
        }
    }
    plot_data = extract_wrapper(file_exts)
    print(plot_data)
    plot_dataset_tokens(plot_data)


def main(args):
    setup_logging()
    logger.info("Starting continue tokenizer training with parameters:")
    logger.info(f"dataset ext: {args.ext}")
    logger.info(f"{args.cont_or_start} training")
    logger.info(f"num threads: {args.num_proc}")
    logger.info(f"tokenizer save dir: {args.save_loc}")
    # logger.info(f"tokenizer save file name: {save_file_name}")
    logger.info(f"num added tokens: {args.added_tokens}")

    old_tokenizer_info = None
    tokenizer = None

    if args.tokenizer_source == "model":
        base_tokenizer_path = f"{args.tokenizer_path_old}/{args.tokenizer_file_old}"
        logger.info(f"base tokenizer path: {base_tokenizer_path}")
        mergeable_ranks = read_tokenizer_from_model(base_tokenizer_path)
    elif args.tokenizer_source == "huggingface":
        base_tokenizer_path = args.tokenizer_path_old
        logger.info(f"Loading tokenizer from huggingface: {base_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
        mergeable_ranks = tokenizer.get_vocab()
        mergeable_ranks.update(tokenizer.get_added_vocab())  # so we dont overwrite the original vocab and added tokens
        old_tokenizer_info = None # TODO save this info
    else:
        raise ValueError(f"Invalid tokenizer source: {args.tokenizer_source}")

    pre_tok = load_pretokenizer(args.pre_tok_name)
    ds = get_data(args.cont_or_start, args.force_retokenize, args.save_tokenized_data, 
                  args.dataset_source_path, args.tokenized_data_name, args.ext, 
                #   args.tokenizer_path_old, args.tokenizer_file_old, 
                  tokenizer, base_tokenizer_path,
                  args.raw_data_name, 
                  pre_tok, args.num_proc, args.batch_size)

    old_info = read_training_info(f"{args.tokenizer_path_old}/{args.info_file_old}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting continue training at: {timestamp:20s}")
    logger.info(f"Is caching enabled: {datasets.is_caching_enabled()}")
    bpe_ranks = mergeable_ranks.copy()

    prev_vocab_size = len(bpe_ranks)
    new_vocab_size = prev_vocab_size + args.added_tokens
    # save_file_name = f"new_mergeable_ranks_{new_vocab_size}"
    # save_file_name = f"info_{new_vocab_size}"

    # ds = ds.select(range(1000))
    # parallelized
    start_timer = time.perf_counter()
    ds_num_tokens = sum(ds["num_tokens"])
    
    data = ds.select_columns("text")
    old_info["current_dataset_size"] = ds_num_tokens

    if "initial_size" not in old_info:
        old_info["initial_size"] = ds_num_tokens

    cache_file_name = f"{args.tokenized_data_name}-{args.ext}.arrow"

    static_info = {
        "pretokenizer_name": args.pre_tok_name,
        # "pretokenizer_string": pre_tok.pre_tokenize_str(),
        "cont_or_start": args.cont_or_start,
        "save_tokenized_data": args.save_tokenized_data,
        "old_tokenzer_info": old_tokenizer_info,
        "base_tokenizer_path": base_tokenizer_path,
    }
    save_module = SaveModule(args.save_loc,
                             original_tokenizer=tokenizer if tokenizer is not None else base_tokenizer_path,
                             original_ds_size=old_info["initial_size"],
                             pretokenizer=pre_tok,
                             static_info=static_info
                             )

    # data = ds.select(range(batch_size)).select_columns("text")
    new_mergeable_ranks, new_merges, dataset_sizes = bpe_continue_train(data, 
                                                                        new_vocab_size, 
                                                                        old_pat_str=None, 
                                                                        initial_ranks=bpe_ranks, 
                                                                        is_pretokenized=args.is_pretokenized, 
                                                                        parallel=args.parallel, 
                                                                        old_data=old_info,
                                                                        cache_file_name=cache_file_name,
                                                                        batch_size=args.batch_size,
                                                                        threads=args.num_proc,
                                                                        save_interval=args.save_interval,
                                                                        save_module=save_module
                                                                        )
    end_timer = time.perf_counter()
    duration = end_timer - start_timer
    compression_rate = dataset_sizes[-1] / old_info['initial_size']

    logger.info(f"Full training run took: {duration:.6f} seconds, and added {len(new_mergeable_ranks) - prev_vocab_size} tokens with a compression rate of {1-compression_rate:.6f}")

    # save_path = f"{args.save_loc}/{save_file_name}"
    # logger.info(f"Saving final tokenizer to {save_path}")
    # save_tokenizer(new_mergeable_ranks, args.save_loc, save_file_name)

    # new_added_tokens = get_new_added_tokens(new_mergeable_ranks, new_merges)

    # training_info = {
    #                 # "state": "final",
    #                 # "merges": new_merges,
    #                 # "new_tokens": new_added_tokens,
    #                 # "base_tokenizer_path": base_tokenizer_path,
    #                 # "number_new_tokens": args.added_tokens,
    #                 # "initial_size": ds_num_tokens,
    #                 # "sizes": dataset_sizes,
    #                 # "save_path": save_path,
    #                 # "duration": duration,
    #                 # "pretokenizer": args.pre_tok_name,
    #                 # "pretokenizer_string": pre_tok.pre_tokenize_str,
    #                 # "cont_or_start": args.cont_or_start,
    #                 # "save_tokenized_data": args.save_tokenized_data,
    #                 # "old_tokenzer_info": old_tokenizer_info,
    #                 # "base_tokenizer": base_tokenizer_path,
    #             }
    
    # save_training_info(training_info, args.save_loc, save_file_name)
    # logger.info(f"Saving info to {args.save_loc}/info_{save_file_name}.json")

    # if tokenizer is not None:
    #     logger.info(f"Saving final tokenizer.json (huggingface) file to {save_path}")
    #     save_tokenizer_new(training_info, args.save_loc, base_tokenizer_path)
    # else:
    #     logger.info(f"Saving final tokenizer.model file to {save_path}")
    #     save_tokenizer(training_info, args.save_loc, save_file_name)

    additional_info = {
        "state": "Final",
        "duration": duration,
        "old_tokenzer_info": old_tokenizer_info,
        "cont_or_start": args.cont_or_start,
    }
    save_module.save(merges=new_merges, dataset_sizes=dataset_sizes, additional_info=additional_info, ranks=new_mergeable_ranks)


def parse_args():

    # added_tokens = 2000
    # batch_size = 500
    # raw_data_name = "genqa"
    # pre_tok_name = "llama3"
    # tokenized_data_name = f"{raw_data_name}_tokenized-{pre_tok_name}"
    # ext = "math"
    # cont_or_start = "cont"
    # force_retokenize = False
    # save_tokenized_data = True

    # # https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
    # # https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json
    # # tokenizer_path_old = "/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer"
    # # tokenizer_file_old = "tokenizer.model"
    # tokenizer_path_old = "/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-code-empty-start"
    # tokenizer_file_old = "new_mergeable_ranks_2000.model"
    # info_file_old = "new_mergeable_ranks_2000.model"

    # save_loc = f"/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-{raw_data_name}-{ext}-{pre_tok_name}-{cont_or_start}"

    try:
        threads = min(psutil.cpu_count(logical=False), len(psutil.Process().cpu_affinity()))
    except:
        threads = os.cpu_count()

    parser = argparse.ArgumentParser(description="Train a tokenizer with BPE")
    
    # Base arguments
    parser.add_argument(
        "--raw-data-name",
        type=str,
        default="genqa",
        help="Name of the raw dataset"
    )

    parser.add_argument(
        "--dataset-source-path",
        type=str,
        default="/fs/cml-projects/llm-pretraining/datasets/raw/genqa/math",
        help="Path to the raw dataset"
    )
    
    # parser.add_argument(
    #     "--dataset-save-path",
    #     type=str,
    #     default="/fs/cml-projects/llm-pretraining/datasets/processed/genqa_tokenized-llama3",
    #     help="Path to save the tokenized dataset"
    # )

    parser.add_argument(
        "--pre-tok-name",
        type=str,
        default="llama3",
        choices=["llama3", "empty"],
        help="Pre-tokenizer to use"
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="Dataset extension/subset to use"
    )

    parser.add_argument(
        "--cont-or-start",
        type=str,
        default="cont",
        choices=["cont", "start"],
        help="Whether to continue training or start fresh"
    )

    parser.add_argument(
        "--added-tokens",
        type=int,
        default=2000,
        help="Number of tokens to add to vocabulary"
    )

    parser.add_argument(
        "--force-retokenize",
        action="store_true",
        help="Force retokenization of data"
    )

    parser.add_argument(
        "--save-tokenized-data",
        action="store_true",
        help="Save tokenized dataset"
    )

    parser.add_argument(
        "--base-tokenizer-path",
        type=str,
        default = None,
        help="Path to base tokenizer"
    )

    parser.add_argument(
        "--tokenized-data-name", 
        type=str,
        default = None
        # default=f"{raw_data_name}_tokenized-{pre_tok_name}"
        )
    
    parser.add_argument(
        "--tokenizer-source",
        type=str,
        default="model",
        choices=["model", "huggingface"],
        help="Source of old tokenizer"
    )
    
    parser.add_argument(
        "--tokenizer-path-old",
        type=str,
        default = "/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer",
        help="Path to old tokenizer"
    )

    parser.add_argument(
        "--tokenizer-file-old",
        type=str,
        default="tokenizer.model",
        help="File name of old tokenizer"
    )

    parser.add_argument(
        "--info-file-old",
        type=str,
        default=None,
        help="File name of old tokenizer info"
    )
    
    parser.add_argument(
        "--save-loc",
        type=str,
        default = None,
        help="Location to save tokenizer"
    )
    
    parser.add_argument(
        "--parallel",
        type=bool,
        choices=[True, False],
        default=True,
        help="Whether to parallelize the training"
    )

    parser.add_argument(
        "--is-pretokenized",
        type=bool,
        choices=[True, False],
        default=True,
        help="Whether the data is pretokenized"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--num-proc",
        type=int,
        default=threads,
        help="Number of threads for parallel processing"
    )

    parser.add_argument(
        "--save-interval",
        type=str,
        default="25",
        help="Number of steps between saving the tokenizer"
    )
    
    # Parse initial args
    args = parser.parse_args()

    save_interval_list = args.save_interval.split(",")
    if len(save_interval_list) == 1:
        args.save_interval = range(0, args.added_tokens, int(args.save_interval))
    else:
        args.save_interval = [int(interval) for interval in save_interval_list]

    # Set dependent defaults
    if not hasattr(args, 'tokenized_data_name') or args.tokenized_data_name is None:
        args.tokenized_data_name = f"{args.raw_data_name}_tokenized-{args.pre_tok_name}"
    
    if not hasattr(args, 'save_loc') or args.save_loc is None:
        args.save_loc = f"tokenizers/Llama-3.2-tokenizer-{args.raw_data_name}{'-' + args.ext if args.ext is not None else ''}-{args.pre_tok_name}-{args.cont_or_start}-{args.added_tokens}"

    if (not hasattr(args, 'info_file_old') or args.info_file_old is None) and args.tokenizer_file_old is not None:
        args.info_file_old = args.tokenizer_file_old

    # if not hasattr(args, 'base_tokenizer_path') or args.base_tokenizer_path is None:
    #     args.base_tokenizer_path = f"tokenizers/Llama-3.2-tokenizer-{args.raw_data_name}-{args.ext}-{args.pre_tok_name}-start"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

