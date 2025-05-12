import json
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from efficient_tokenization.tokenization_utils import SaveModule
import os

def load_tokenizer_json_correctly(tokenizer_path):

    tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
    tokenizer_name = tokenizer_path.split("/")[-1]
    print(f"Loading tokenizer from {tokenizer_json_path}")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Step 1: Merge vocab + added_tokens manually
    print("Merging vocab + added_tokens")
    base_vocab = tokenizer_data["model"]["vocab"]
    added_tokens = tokenizer_data.get("added_tokens", [])

    merged_vocab = {**base_vocab}
    for token in added_tokens:
        merged_vocab[token["content"]] = token["id"]

    # Step 2: Build BPE model from merged vocab
    print("Building BPE model from merged vocab")
    merges = tokenizer_data["model"].get("merges", [])
    merges = [tuple(merge) for merge in merges]
    bpe_model = models.BPE(vocab=merged_vocab, merges=merges, dropout=None, unk_token=None)

    tokenizer = Tokenizer(bpe_model)

    # Step 3: Rebuild pre-tokenizer, decoder, etc (same as before)
    print("Rebuilding pre-tokenizer, decoder, etc")
    pretokenizer_info = tokenizer_data.get("pre_tokenizer", {})
    if pretokenizer_info.get("type") == "ByteLevel":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=pretokenizer_info.get("add_prefix_space", False),
            trim_offsets=pretokenizer_info.get("trim_offsets", True),
            use_regex=pretokenizer_info.get("use_regex", True)
        )

    decoder_info = tokenizer_data.get("decoder", {})
    if decoder_info.get("type") == "ByteLevel":
        tokenizer.decoder = decoders.ByteLevel(
            add_prefix_space=decoder_info.get("add_prefix_space", True),
            trim_offsets=decoder_info.get("trim_offsets", True),
            use_regex=decoder_info.get("use_regex", True)
        )

    # Step 4: Wrap into PreTrainedTokenizerFast
    print("Wrapping into PreTrainedTokenizerFast")
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    hf_tokenizer.save_pretrained(f"/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/modified/{tokenizer_name}")


    return hf_tokenizer

def load_tokenizer_json_correctly2(original_tokenizer_path: str, output_suffix: str = ""):
    
    # Code to convert extended tokenizer to added tokens
    # base_tokenizer_path = "microsoft/phi-4-mini-reasoning"
    # original_tokenizer_path = "tokenizers/phi_gsm8k_cot-1000"
    print(f"Loading tokenizer from {original_tokenizer_path}")
    # tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)

    with open(os.path.join(original_tokenizer_path, "training_info.json"), "r") as f:
        tokenizer_json = json.load(f)

    merges = tokenizer_json["merges"]
    sizes = tokenizer_json["sizes"]

    tokenizer_name = original_tokenizer_path.split("/")[-1]
    num_new_tokens = int(tokenizer_name.split("-")[-1])

    path = original_tokenizer_path + output_suffix

    sm = SaveModule.from_path(original_tokenizer_path)

    sm.shrink_tokenizer(num_new_tokens, merges, sizes, new_path=path, added_tokens=True)



if __name__ == "__main__":
    tokenizer_name_list = [
        "phi_gsm8k_cot-10",
        "phi_gsm8k_cot-50",
        "phi_gsm8k_cot-100",
        "phi_gsm8k_cot-1000",
    ]
    for tokenizer_name in tokenizer_name_list:
        print(f"Processing {tokenizer_name}")
        tokenizer = load_tokenizer_json_correctly2(f"/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/{tokenizer_name}")


    # # 🔥 Test that your reserved_special_tokens are intact
    # print(f"tokenizer.convert_ids_to_tokens(128034) should be <|reserved_special_token_26|> : {tokenizer.convert_ids_to_tokens(128034)}")  # Should print <|reserved_special_token_26|>
    # print(f"tokenizer.convert_tokens_to_ids('<|reserved_special_token_26|>') should be 128034: {tokenizer.convert_tokens_to_ids('<|reserved_special_token_26|>')}")  # Should print 128034
    # print(f"tokenizer.convert_ids_to_tokens(129034) should be 'ĠrefersĠtoĠthe': {tokenizer.convert_ids_to_tokens(129034)}") # Should print "ĠrefersĠtoĠthe"
    # print(f"tokenizer.convert_tokens_to_ids('ĠrefersĠtoĠthe') should be 129034: {tokenizer.convert_tokens_to_ids('ĠrefersĠtoĠthe')}") # Should print 129034
