import argparse
from efficient_tokenization.tokenization_utils import SaveModule, get_new_path
from transformers import AutoTokenizer
import os
import json


def run_tokenizer_shrink(args):

    original_tokenizer_path = args.old_tokenizer_path
    print(f"Loading tokenizer from {original_tokenizer_path}")
    sm = SaveModule.from_path(original_tokenizer_path)

    # tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)
    # print(f"Loaded HF tokenizer from vocab file: {original_tokenizer_path}")

    with open(os.path.join(args.old_tokenizer_path, "training_info.json"), "r") as f:
        tokenizer_json = json.load(f)

    merges = tokenizer_json["merges"]
    sizes = tokenizer_json["sizes"]

    # additional_info = tokenizer_json["additional_info"]
    num_new_tokens_list = args.num_new_tokens_list
    for num_new_tokens in num_new_tokens_list:
        path = get_new_path(num_new_tokens, original_tokenizer_path)
        sm.shrink_tokenizer(num_new_tokens, merges, sizes, new_path=path, added_tokens=args.add_special_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-tokenizer-path", type=str, required=True)
    # parser.add_argument("--new-tokenizer-path", type=str, required=True)
    parser.add_argument("--num-new-tokens-list", type=str, required=True, default="1,5,20,50,100,200,300,400,500,600,700,800,900")
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()
    args.num_new_tokens_list = [int(x) for x in args.num_new_tokens_list.split(",")]
    run_tokenizer_shrink(args)
