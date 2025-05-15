from efficient_tokenization.extend_embeddings import extend_model_embeddings
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import argparse
import os
from safetensors import safe_open
import shlex

import json
from pathlib import Path



def patch_lm_script(model_path_parent: str) -> str:
    with open(os.path.join(model_path_parent, "lm_eval.sh"), "r") as f:
        lines = f.readlines()
    
    assert len(lines) == 1, "Only one line is allowed in the lm_eval.sh file"
    cmd = lines[0]
    
    # Tokenize like a shell would
    tokens = shlex.split(cmd)

    def extract_flag_dict(flag, tokens):
        if flag in tokens:
            value = tokens[tokens.index(flag) + 1]
            return dict(pair.split("=", 1) for pair in value.split(","))
        return {}

    def replace_flag_dict(flag, tokens, new_values):
        if flag in tokens:
            i = tokens.index(flag) + 1
            updated = ",".join(f"{k}={v}" for k, v in new_values.items())
            tokens[i] = updated
        else:
            tokens += [flag, ",".join(f"{k}={v}" for k, v in new_values.items())]
        return tokens

    model_args_dict = extract_flag_dict("--model_args", tokens)
    extra_config_dict = extract_flag_dict("--extra_config", tokens)

    # print("Model Args:", model_args_dict)
    # print("Extra Config:", extra_config_dict)
    train_info_dict = json.load(open(os.path.join(model_path_parent, "train_config.json"), "r"))

    base_model_name = "Llama-3.2-3B"
    new_model_name = "Llama-3.2-3B-Instruct"

    pretrained = train_info_dict.get("model_name", None)
    if not pretrained == base_model_name:
        return False, f"Skipping {model_path_parent} because it is not created from base path"
    
    embeddings_path = model_args_dict.get("embeddings", extra_config_dict.get("embeddings", None))

    if embeddings_path is None:
        embeddings_path = os.path.join(model_path_parent, "final_model", "embeddings_only.pt")
        if not os.path.exists(embeddings_path):
            return False, f"No embeddings path found for {model_path_parent}"
    
    model_args_dict["pretrained"] =f"meta-llama/{new_model_name}"
    if "embeddings" in model_args_dict:
        del model_args_dict["embeddings"]

    extra_config_dict["embeddings"] = embeddings_path

    # if train_info_dict.get("finetuning_params", None) == "new_tokens_only":
    #     extra_config_dict["new_only"] = True
    # elif train_info_dict.get("finetuning_params", None) == "embeddings":
    #     pass
    # else:
    #     return False, f"finetuning: {train_info_dict.get('finetuning_params', None)} doesnt include patched embeddings"

    if train_info_dict.get("finetuning_params", None) == "new_tokens_only" or train_info_dict.get("finetuning_params", None) == "embeddings":
        extra_config_dict["new_only"] = True
    else:
        return False, f"finetuning: {train_info_dict.get('finetuning_params', None)} doesnt include patched embeddings"

    tokens = replace_flag_dict("--extra_config", tokens, extra_config_dict)
    tokens = replace_flag_dict("--model_args", tokens, model_args_dict)

    new_cmd = shlex.join(tokens)
    new_cmd = new_cmd.replace("new_only=None", "new_only=True")
    new_cmd = new_cmd.replace("do_sample=False,temperature=None,top_p=None", "do_sample=False,temperature=0.0,top_p=1.0")
    return True, new_cmd


def convert_all_models_to_embedddings(experiment_path):
    all_commands = []
    if os.path.isdir(experiment_path):
        for model_name in os.listdir(experiment_path):
            model_path = os.path.join(experiment_path, model_name)
            model_path_final = os.path.join(model_path, "final_model")
            if os.path.isdir(model_path_final):
                if os.path.exists(os.path.join(model_path_final, "embeddings_only.pt")):
                    # print(f"Skipping {model_path} because it already has embeddings")
                    pass
                else:
                    # print(f"Converting {model_path} to embeddings")
                    success = get_embeddings_only(model_path_final)
                    if not success:
                        print(f"Failed to get embeddings for {model_path}")
                        return
            else:
                print(f"Skipping {model_path} because it is not a directory")
                
            success, output_str = patch_lm_script(model_path)
            if not success:
                print(f"{model_name}: {output_str}")
            else:
                print(f"{model_name}: {success}")
                all_commands.append(output_str)
    else:
        print(f"Experiment path {experiment_path} is not a directory")

    with open(os.path.join("scripts", "compiled_eval_scripts.sh"), "a") as f:
        f.write("\n# PATCHED\n")
        for command in all_commands:
            f.write(f"{command}\n")

def get_embeddings_only(model_path):
    # Load all safetensors files from directory or use single file
    if os.path.isdir(model_path):
        safetensor_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".safetensors")]
        )
    else:
        safetensor_files = [model_path]

    if len(safetensor_files) == 0:
        print(f"{model_path}: No safetensors files found")
        return False

    param_dict = {}
    for file in safetensor_files:
        with safe_open(file, framework="pt") as f:
            for key in f.keys():
                param_dict[key] = f.get_tensor(key)


    try:
        input_embeddings = param_dict["model.embed_tokens.weight"].cpu()
    except:
        print(f"{model_path}: model.embed_tokens.weight not found in safetensors files. Options: {param_dict.keys()}")
        return False

    try:
        outpt_embeddings = param_dict["lm_head.weight"].cpu()
    except:
        # print(f"{model_path}: lm_head.weight not found in safetensors files. Options: {param_dict.keys()}")
        outpt_embeddings = input_embeddings.clone()
    
    embeddings_path = os.path.join(model_path, "embeddings_only.pt")

    # better way to do it
    # model = LlamaForCausalLM.from_pretrained(model_path)
    # input_embeddings = model.get_input_embeddings().weight.detach().cpu()
    # output_embeddings = model.get_output_embeddings().weight.detach().cpu()

    torch.save({
        "input_embeddings": input_embeddings,
        "output_embeddings": outpt_embeddings,
    }, embeddings_path)
    # print(f"Saved embeddings to {embeddings_path}")
    return True


def main(args):
    if args.tokenizer_path is not None:
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None

    print(f"Loading model from {args.model_path}")
    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Patching model embeddings using strategy {args.embedding_init_strategy}")
    model = extend_model_embeddings(
        model,
        args.num_new_tokens,
        init_strategy=args.embedding_init_strategy,
        tokenizer=tokenizer,
        import_path=args.import_path
    )

    output_folder_splitted = args.import_path.split("/")
    output_folder = "/".join(output_folder_splitted[:-1])
    input_model_name = output_folder_splitted[-2]
    if args.output_name is None:
        model_name = args.model_path.split("/")[-1]
        output_name = f"{model_name}_plus_{input_model_name}"
    else:
        output_name = args.output_name

    if args.experiment_name is not None:
        output_folder = f"output/{args.experiment_name}"

    output_path = os.path.join(output_folder, output_name)
    
    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--num-new-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--embedding-init-strategy", type=str, default="import")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--save-embeddings-only", action="store_true")
    parser.add_argument("--save-embeddings-experiment", action="store_true")
    args = parser.parse_args()

    # model_path = "meta-llama/Llama-3.2-3B-Instruct"
    # num_new_tokens = 1000
    # tokenizer_path = "/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000"
    model = "/cmlscratch/astein0/efficient_tokenization_for_inference/output/full_patching/0a4e765d-Llama-3.2-3B-Instruct-mixed-500/final_model"
    if args.save_embeddings_only:
        get_embeddings_only(args.model_path)
    elif args.save_embeddings_experiment:
        convert_all_models_to_embedddings(os.path.join("output", args.experiment_name))
    else:
        main(args)