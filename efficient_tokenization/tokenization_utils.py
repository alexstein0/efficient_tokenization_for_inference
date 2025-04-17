from typing import Dict, Any, List, Tuple
import os
import json
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import pre_tokenizers, Regex
from datasets import Dataset
import base64

from efficient_tokenization.tokenize_simple import pretokenizer_to_config, load_pretokenizer



def save_tokenizer_to_huggingface(original_tokenizer: PreTrainedTokenizerFast, new_tokenizer_path: str, new_tokenizer_json: dict):
    # 2️⃣ Define new save path
    os.makedirs(new_tokenizer_path, exist_ok=True)

    # 3️⃣ Save the tokenizer to a directory
    original_tokenizer.save_pretrained(new_tokenizer_path)

    tokenizer_json_path = os.path.join(new_tokenizer_path, "tokenizer.json")
    with open(tokenizer_json_path, "w") as f:
        json.dump(new_tokenizer_json, f, indent=2)


def read_tokenizer_from_model(path: str):
    with open(path, 'rb') as f:
        contents = f.read()
        contents = {
            token: rank
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }
        mergeable_ranks = {
            # base64.b64decode(token): int(rank)
            token_bytes_to_string(base64.b64decode(token)): int(rank)
            for token, rank in contents.items()
        }
    return mergeable_ranks

def save_tokenizer(ranks_to_save, save_dir: str, save_name: str = None):
    os.makedirs(save_dir, exist_ok=True)
    save_name = save_name if save_name is not None else "tokenizer"

    # Save new_mergeable_ranks to a file in the same format as it was loaded
    with open(f'{save_dir}/{save_name}.model', 'wb') as f:
        for token, rank in ranks_to_save.items():
            # Encode the token to base64
            encoded_token = base64.b64encode(string_to_token_bytes(token)).decode('utf-8')
            # print(f"rank: {rank}, token: {token}, type: {type(token)}, encoded: {encoded_token}, type: {type(encoded_token)}")
            f.write(f"{encoded_token} {rank}\n".encode('utf-8'))


def save_training_info(data_dict: Dict[str, Any], save_dir: str, save_name: str = "training_info"):
    os.makedirs(save_dir, exist_ok=True)
    # Save data_dict to a file in the same format as it was loaded
    with open(f'{save_dir}/{save_name}.json', 'w') as f:
        json.dump(data_dict, f, indent = 4)


def read_training_info(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        print(f"Error loading training info from {path}")
        return {}

def save_tokenizer_new(info_dict: Dict[str, Any], save_dir: str, prior_tokenizer: str | PreTrainedTokenizerFast, pretok_override = None):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving tokenizer to {save_dir}")
    if isinstance(prior_tokenizer, str):
        prior_tokenizer = AutoTokenizer.from_pretrained(prior_tokenizer)
    new_tokenizer_info = convert_tokenizer_to_huggingface_correct(info_dict, prior_tokenizer, pretok_override)
    save_tokenizer_to_huggingface(prior_tokenizer, save_dir, new_tokenizer_info)

### Read and save tokenizer in .model files


def convert_tokenizer_to_huggingface_correct(new_tokenizer_info: Dict[str, Any], original_tokenizer: PreTrainedTokenizerFast, pretok_override = None):
    # must get original tokenizer from huggingface
    tokenizer_json = json.loads(original_tokenizer._tokenizer.to_str())
    
    # old_vocab = tokenizer_json["model"]["vocab"]
    # starting_index = len(tokenizer.get_vocab())
    old_merges = tokenizer_json["model"]["merges"]

    # Extract vocab (token: index)
    old_vocab = original_tokenizer.get_vocab()
    
    # load new_tokenizer_info_path
    # new_tokenizer_info = read_training_info(new_tokenizer_info_path)
    # get merges and new_tokens
    new_merges = new_tokenizer_info["merges"]
    new_tokens = new_tokenizer_info["new_tokens"]

    # Update vocab (append at the next available ID)
    new_vocab = {**old_vocab}  # Copy the old vocab
    starting_index = max(old_vocab.values()) + 1

    for i, token in enumerate(new_tokens):
        new_vocab[token] = starting_index + i

    new_vocab_sorted = dict(sorted(new_vocab.items(), key=lambda item: item[1]))
    # new_vocab_sorted = new_vocab

    joined_merges = [x for x in old_merges]
    joined_merges.extend(new_merges)

    added_tokens = original_tokenizer.get_added_vocab()
    add_tok_ids = [tok_id for _, tok_id in added_tokens.items()]

    new_vocab_sorted_no_added = {tok: tok_id for tok, tok_id in new_vocab_sorted.items() if tok_id not in add_tok_ids}

    new_tokenizer_json = {**tokenizer_json}
    new_tokenizer_json["model"]["vocab"] = new_vocab_sorted_no_added
    new_tokenizer_json["model"]["merges"] = joined_merges
    if pretok_override is not None:
        pre_tok_config = pretokenizer_to_config(pretok_override)
        new_tokenizer_json["pre_tokenizer"] = pre_tok_config

    # add new tokens to tokenizer
    return new_tokenizer_json

def get_new_added_tokens(new_merges: List[Tuple[bytes, bytes]], new_mergeable_ranks: Dict[bytes, int] = None) -> List[bytes]:
    new_added_tokens = []
    for merge in new_merges:
        final = merge[0]+merge[1]
        new_added_tokens.append(final)

        if new_mergeable_ranks is not None:
            rank_id1 = new_mergeable_ranks[merge[0]]
            rank_id2 = new_mergeable_ranks[merge[1]]
            final_id = new_mergeable_ranks[final]
            # print(f"{merge[0]}:{rank_id1} {merge[1]}:{rank_id2} {final}:{final_id}")

    return new_added_tokens

def get_new_path(num_new_tokens: int, old_path: str):
    path, name = ("/".join(old_path.split("/")[:-1]), old_path.split("/")[-1])
    new_name, old_token_amount = ("-".join(name.split("-")[:-1]), name.split("-")[-1])
    old_token_amount = int(old_token_amount)
    new_name = f"{new_name}-{num_new_tokens}"
    new_path = f"{path}/{new_name}"
    
    return new_path

class SaveModule:
    def __init__(self, save_loc: str, original_tokenizer: PreTrainedTokenizerFast | str, original_ds_size: int, pretokenizer: pre_tokenizers.PreTokenizer | str, static_info: Dict[str, Any] = {}):
        self.save_loc = save_loc
        if isinstance(original_tokenizer, str):
            self.original_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer)
        else:
            self.original_tokenizer = original_tokenizer

        self.initial_vocab_size = len(self.original_tokenizer.get_vocab())
        self.original_ds_size = original_ds_size
        
        self.static_info = {**static_info,
            "save_path": self.save_loc,
            "original_dataset_size": self.original_ds_size,
            "initial_vocab_size": self.initial_vocab_size,
        }

        if isinstance(pretokenizer, str):
            self.pretokenizer = load_pretokenizer(pretokenizer)
        else:
            self.pretokenizer = pretokenizer


    @classmethod
    def from_path(cls, old_path: str, new_path: str = None):
        # only applicable if you are saving to new tokenizer
        training_info = read_training_info(os.path.join(old_path, "training_info.json"))
        static_info = training_info.get("static_info", {})
        static_info["created_from_path"] = old_path
        print(static_info)
        # TODO save_loc can be set later
        return cls(save_loc=new_path, original_tokenizer=static_info["base_tokenizer_path"], original_ds_size=static_info["original_dataset_size"], pretokenizer=static_info["pretokenizer_name"], static_info=static_info)

    def shrink_tokenizer(self, num_new_tokens: int, merges: List[Tuple[bytes, bytes]], dataset_sizes: List[int], additional_info: Dict[str, Any] = {}, ranks: Dict[bytes, int] = None, new_path: str = None):
        assert num_new_tokens is not None, "num_new_tokens must be provided"
        assert num_new_tokens < len(merges), "num_new_tokens must be less than the original increase in vocab size"
        new_merges = merges[:num_new_tokens]
        new_dataset_sizes = dataset_sizes[:num_new_tokens]
        new_ranks = None
        
        if ranks is not None:
            new_ranks = {}
            for i, (k, v) in enumerate(ranks.items()):
                if i < num_new_tokens:
                    new_ranks[k] = v
        additional_info["state"] = "Final_(shrunk)"
        if new_path is not None:
            self.save_loc = new_path
            print(f"Setting save_loc to {self.save_loc}")

        self.save(merges=new_merges, dataset_sizes=new_dataset_sizes, additional_info=additional_info, ranks=new_ranks, num_added_tokens=num_new_tokens)

    def save(self, merges: List[Tuple[bytes, bytes]], dataset_sizes: List[int], additional_info: Dict[str, Any] = {}, ranks: Dict[bytes, int] = None, num_added_tokens: int = None):
        new_added_tokens = get_new_added_tokens(new_merges=merges, new_mergeable_ranks=ranks)
        if ranks is not None:
            num_added_tokens = len(ranks) - self.initial_vocab_size
        elif num_added_tokens is None:
            raise ValueError("No new added tokens or ranks provided")
        
        compression_rate = 1 - (dataset_sizes[-1]/self.original_ds_size)
        new_training_info = {
            # "state": step,
            "merges": merges,
            "new_tokens": new_added_tokens,
            "number_new_tokens": num_added_tokens,
            "sizes": dataset_sizes,
            # "duration": "0",
            # "cont_or_start": args.cont_or_start,
            # "save_tokenized_data": args.save_tokenized_data,
            # "old_tokenzer_info": old_tokenizer_info
            "compression_rate": compression_rate
        }
        self.static_info["save_path"] = self.save_loc
        training_info = {"static_info": self.static_info, **additional_info, **new_training_info}

        if self.original_tokenizer is not None:
            # will never be none, better check wll be f we allow the .model files to be used
            save_tokenizer_new(training_info, self.save_loc, self.original_tokenizer, pretok_override=self.pretokenizer)
        else:
            save_tokenizer(training_info, self.save_loc)

        save_training_info(training_info, self.save_loc)


# def cut_tokenizer_to_size(tokenizer: PreTrainedTokenizerFast, size: int):


#### RENDERING STUFF

### Handle rendering of specific bytes into visable characters

# copied from https://github.com/huggingface/transformers/pull/30334/files#diff-08a7e5c7b50f73fc176e9a35899810080f0bc5b9e54278866f2b48ce68ddca30R1491

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()

def token_bytes_to_string(b):
    return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])



def unicode_to_bytes():
    """
    Returns a mapping from unicode strings back to their original utf-8 bytes.
    This reverses the `bytes_to_unicode` mapping.
    """
    # byte_encoder = bytes_to_unicode()  # Original byte-to-unicode mapping
    return {v: k for k, v in byte_encoder.items()}

byte_decoder = unicode_to_bytes()

def string_to_token_bytes(s):
    """
    Converts a string back into token bytes using the reverse mapping.

    Args:
        s (str): The input string to convert.

    Returns:
        bytes: The byte representation of the string.
    """
    return bytes([byte_decoder[char] for char in s])


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


from typing import Dict
def compare_dicts(dict1: Dict, dict2: Dict) -> bool:
    """
    Compare two dictionaries and print differences if they exist.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        bool: True if dictionaries are identical, False otherwise
    """
    if dict1.keys() != dict2.keys():
        print("Different keys:")
        print("Keys only in first dict:", set(dict1.keys()) - set(dict2.keys()))
        print("Keys only in second dict:", set(dict2.keys()) - set(dict1.keys()))
        return False
    
    differences = {
        k: (dict1[k], dict2[k])
        for k in dict1
        if dict1[k] != dict2[k]
    }
    
    if differences:
        print("Different values:")
        for k, (v1, v2) in differences.items():
            print(f"Key: {k}")
            print(f"  Dict1: {v1}")
            print(f"  Dict2: {v2}")
        return False
        
    return True
