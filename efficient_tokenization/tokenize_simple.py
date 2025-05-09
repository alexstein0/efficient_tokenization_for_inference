from tokenizers import pre_tokenizers, Regex
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from typing import Dict, List, Optional
# import logging
import psutil
from datasets import load_dataset, load_from_disk
import datasets
datasets.disable_caching()
# from chat_templating_old import apply_chat_template_to_repeat, get_llama_base_chat_template, get_llama_instruct_chat_template
from chat_templating import apply_chat_template_repeat, apply_chat_template_normal, get_llama32_instruct_chat_template, get_llama32_repeat_chat_template
import os
import copy

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


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
        output.append(" ".join(raw_messages))
    return output

def extract_magpie(example_list: List[List[List[Dict[str, str]]]], track_role: bool = False, flattened: bool = False) -> List[List[str]]:
    output = []
    role_map = {
        'human': 'user',
        'gpt': 'assistant',
        'system': 'system',
    }
    if flattened:
        for text in example_list:
            raw_messages = []
            for message in text:
                this_message = ""
                if track_role and message.get('from') is not None:
                    from_role = role_map[message.get('from')]
                    this_message += f'<|{from_role}|> '
                content = message.get('value', '').strip()
                this_message += content
                raw_messages.append(this_message.strip())
            # raw_messages = [message.get('content', '').strip() for message in text if message.get('content')]
            output.append(" ".join(raw_messages))
    else:
        for conversation in example_list:
            out_conversation = []
            for message in conversation:
                out_message = {}
                if track_role and message.get('from') is not None:
                    out_message['role'] = role_map[message.get('from')]
                out_message['content'] = message.get('value', '').strip()
                out_conversation.append(out_message)
            output.append(out_conversation)
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

def flatten_magpie_conversations(example: Dict[str, List[List[Dict[str, str]]]],
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

    text_col = example.get('conversations', [])
    extracted_texts = extract_magpie(text_col, track_role, flattened=False)

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
    # for example_text in extracted_texts:
    if 1:
        # raw_messages = [message.get('content', '').strip() for message in text if message.get('content')]
        tokenized_messages = []
        # for i, message in enumerate(example_text):
        for i, message in enumerate(extracted_texts):
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
    # from tiktoken
    # pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # whitespace_str = r"""\s+(?!\S)|\s+"""
    # empty_regex = r'.*'
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

def pretokenizer_to_config(pre_tok):
    """Convert any pretokenizer to a config dictionary by parsing its string representation."""
    str_rep = str(pre_tok)
    
    def parse_params(param_str):
        """Parse parameters from string representation into a dictionary."""
        params = {}
        # Handle empty parameters
        if not param_str:
            return params
            
        for param in param_str.split(", "):
            key, value = param.split("=")
            # Try to convert value to appropriate type
            if value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            elif value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '').isdigit():
                params[key] = float(value)
            else:
                # Remove quotes if present
                params[key] = value.strip("'\"")
        return params
    
    def parse_pretokenizer(tok_str):
        # VIA cursor
        """Recursively parse a pretokenizer string into a config dictionary."""
        # Find the type and parameters
        tok_type = tok_str[:tok_str.find("(")]
        param_str = tok_str[tok_str.find("(")+1:tok_str.rfind(")")]
        
        # Handle nested pretokenizers (like in Sequence)
        if tok_type == "Sequence":
            # Extract the list of pretokenizers
            pretok_list = param_str[len("pretokenizers=["):-1]
            # Split on "), " but keep the closing parenthesis
            nested_toks = []
            current = ""
            paren_count = 0
            for char in pretok_list:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                current += char
                if paren_count == 0 and char == ")":
                    nested_toks.append(current)
                    current = ""
                elif paren_count == 0 and char == ",":
                    current = ""
                
            return {
                "type": tok_type,
                "pretokenizers": [parse_pretokenizer(tok.strip()) for tok in nested_toks if tok.strip()]
            }
        else:
            # Regular pretokenizer
            return {
                "type": tok_type,
                **parse_params(param_str)
            }
    
    return parse_pretokenizer(str_rep)

def load_tokenizer_from_file(vocab_file_path: str, base_tokenizer: AutoTokenizer = None):
    if base_tokenizer is not None:
        tok = PreTrainedTokenizerFast(
            vocab_file=vocab_file_path,
            bos_token=base_tokenizer.bos_token,
            eos_token=base_tokenizer.eos_token,
            unk_token=base_tokenizer.unk_token,
            sep_token=base_tokenizer.sep_token,
            pad_token=base_tokenizer.pad_token,
            cls_token=base_tokenizer.cls_token,
            mask_token=base_tokenizer.mask_token,
            additional_special_tokens=base_tokenizer.additional_special_tokens,
            # Copy other important attributes
            clean_up_tokenization_spaces=base_tokenizer.clean_up_tokenization_spaces,
            model_max_length=base_tokenizer.model_max_length,
            padding_side=base_tokenizer.padding_side,
            truncation_side=base_tokenizer.truncation_side
            )
        # tok.special_tokens_map = base_tokenizer.special_tokens_map.copy()

    else:
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

# def get_tokenized_data(tokenizer_path: str, dataset_path: str, pre_tok_name: Optional[str] = None):
#     tok = load_tokenizer_from_file(tokenizer_path)
#     if pre_tok_name is not None:
#         tok = apply_pretokenizer(tok, pre_tok_name)
#     # logging.info(f"Tokenizing using {tokenizer_path}")
#     ds = load_from_disk(dataset_path)
#     ds = get_genqa_data(ds)
    
#     tokenized_dataset = my_tokenize(ds.select_columns("text"), tok)

#     return tok, ds, tokenized_dataset

def get_tokenizer(tokenizer_path: str, pre_tok_name: str = None, old_tokenizer: AutoTokenizer = None):
    tok = load_tokenizer_from_file(tokenizer_path, old_tokenizer)
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

def get_magpie_data(ds: datasets.Dataset, tokenizer = None, track_role: bool = False, threads=1, batch_size=1000):
    fn_kwargs={
        "tokenizer": tokenizer,
        "track_role": track_role,
        }
    ds = ds.select_columns("conversations").map(
        flatten_magpie_conversations,
        num_proc=threads,
        batched=True,
        batch_size=batch_size,
        fn_kwargs=fn_kwargs,
        )
    return ds


def init_tokenizer(tokenizer_path):
    from transformers import AutoTokenizer
    global extended_tokenizer
    extended_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# def create_translation_dataset(ds: datasets.Dataset, base_tokenizer: AutoTokenizer, tokenizer_input: AutoTokenizer | str, batch_size: int, threads: int) -> datasets.Dataset:
#     meta_prompt = """Repeat the following text: """
#     meta_prompt_tok = base_tokenizer(meta_prompt)

#     text1_label = "text1: "
#     text1_label_tok = base_tokenizer(text1_label, add_special_tokens=False)
#     text2_label = "text2: "
#     text2_label_tok = base_tokenizer(text2_label, add_special_tokens=False)
    
#     def tokenize_function(examples):
#         global extended_tokenizer
#         batch_size = len(examples["text"])
#         row_texts = list(examples["text"])
        
#         text_tok1 = base_tokenizer(row_texts, add_special_tokens=False)
#         text_tok2 = extended_tokenizer(row_texts, add_special_tokens=False)
        
#         combined_text = []
#         combined_input_ids = []
#         combined_attention_mask = []
#         combined_loss_mask = []
        
#         for i in range(batch_size):
#             text = (meta_prompt + 
#                     text1_label + 
#                     row_texts[i] + 
#                     text2_label +
#                     row_texts[i]
#                     )
#             input_ids = (
#                 meta_prompt_tok['input_ids'] + 
#                 text1_label_tok['input_ids'] +
#                 text_tok1['input_ids'][i] +
#                 text2_label_tok['input_ids'] + 
#                 text_tok2['input_ids'][i]
#             )
#             attention_mask = (
#                 meta_prompt_tok['attention_mask'] + 
#                 text1_label_tok['attention_mask'] +
#                 text_tok1['attention_mask'][i] +
#                 text2_label_tok['attention_mask'] +
#                 text_tok2['attention_mask'][i]
#             )
#             mask_loss = (
#                 [0 for _ in meta_prompt_tok['attention_mask']] + 
#                 [0 for _ in text1_label_tok['attention_mask']] +
#                 [0 for _ in text_tok1['attention_mask'][i]] +
#                 [0 for _ in text2_label_tok['attention_mask']] +
#                 [1 for _ in text_tok2['attention_mask'][i]]
#             )
            
#             combined_text.append(text)
#             combined_input_ids.append(input_ids)
#             combined_attention_mask.append(attention_mask)
#             combined_loss_mask.append(mask_loss)

#         return {
#             "text": combined_text,
#             "input_ids": combined_input_ids,
#             "attention_mask": combined_attention_mask,
#             "labels": combined_input_ids.copy(),
#             "loss_mask": combined_loss_mask,
#         }

#     if isinstance(tokenizer_input, str):
#         # Initialize the tokenizer before mapping
#         init_tokenizer(tokenizer_input)
#     else:
#         # When using an object, make it available globally
#         global extended_tokenizer
#         extended_tokenizer = tokenizer_input
        
#     tokenized_dataset = ds.map(
#         tokenize_function,
#         batched=True,
#         batch_size=batch_size,
#         num_proc=threads,
#         remove_columns=ds.column_names,
#         desc="Creating translation dataset",
#         cache_file_name=None
#     )
    
#     return tokenized_dataset

def create_translation_dataset_with_template(ds: datasets.Dataset, base_tokenizer: AutoTokenizer, tokenizer_input: AutoTokenizer | str, batch_size: int, threads: int) -> datasets.Dataset:
    chat_template = get_llama32_repeat_chat_template()
    def tokenize_function(examples):
        tokenized_texts = apply_chat_template_repeat(
            base_tokenizer=base_tokenizer,
            second_tokenizer=tokenizer_input,
            conversation=examples["text"],
            chat_template=chat_template,
            # return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return tokenized_texts

    # if isinstance(tokenizer_input, str):
    #     # Initialize the tokenizer before mapping
    #     init_tokenizer(tokenizer_input)
    # else:
    #     # When using an object, make it available globally
    #     global extended_tokenizer
    #     extended_tokenizer = tokenizer_input
        
    tokenized_dataset = ds.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=threads,
        remove_columns=ds.column_names,
        desc="Creating translation dataset",
        cache_file_name=None
    )
    
    return tokenized_dataset

def apply_chat_template(ds: datasets.Dataset, tokenizer: AutoTokenizer, batch_size: int, threads: int) -> datasets.Dataset:
    chat_template = get_llama32_instruct_chat_template()
    def tokenize_function(examples):
        tokenized_texts = apply_chat_template_normal(
            tokenizer=tokenizer,
            conversation=examples["text"],
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            chat_template=chat_template, # can change to base template
            tokenize=True,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return tokenized_texts
        # return {
        #     "input_ids": tokenized_texts["input_ids"], 
        #     "loss_mask": tokenized_texts["loss_mask"],
        #     "attention_mask": tokenized_texts["attention_mask"],
        #     "labels": tokenized_texts["input_ids"].clone(),
        #     }
    
    tokenized_dataset = ds.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=threads,
        remove_columns=ds.column_names,
        desc="Creating translation dataset",
        cache_file_name=None
    )
    
    return tokenized_dataset


if __name__ == "__main__":

    raw_data_name = "genqa"
    pre_tok_name = "empty"
    ext = "math"


    tokenizer_path_old = f"/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-{ext}-{pre_tok_name}-start"
    tokenizer_file_old = "new_mergeable_ranks_2000.model"
    vocab_file_path = f"{tokenizer_path_old}/{tokenizer_file_old}"
    
    ds_path = f"/fs/cml-projects/llm-pretraining/datasets/raw/{raw_data_name}/{ext}"

    # tok, ds = get_tokenized_data(vocab_file_path, ds_path, pre_tok_name=pre_tok_name)

