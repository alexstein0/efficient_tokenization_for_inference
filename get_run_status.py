from efficient_tokenization.utils import get_latest_checkpoint
from efficient_tokenization.benchmarking_utils import parse_args_from_file, convert_argparse_to_hash_path, get_lm_eval_string
import argparse
import torch
import os


def has_safetensors_files(directory):
    """Check if any files in the directory end with '.safetensors'."""
    try:
        files = os.listdir(directory)
        for file in files:
            if file.endswith('.safetensors'):
                return True
        return False
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return False


def get_run_status(output_dir: str) -> str:
    if os.path.exists(os.path.join(output_dir, "final_model")):
        directory_path = os.path.join(output_dir, "final_model")
        if has_safetensors_files(directory_path):
            return "Final model exists"
        else:
            print(f"No files ending with '.safetensors' found in the directory {directory_path}.")
    if os.path.exists(os.path.join(output_dir, "checkpoints")):
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint is not None:
            state_dict_path = os.path.join(latest_checkpoint, "checkpoint_meta.pt")
            if not os.path.exists(state_dict_path):
                return "Training corrupted, no checkpoint_meta.pt found"
            train_info = torch.load(state_dict_path)
            return f"Latest checkpoint at {train_info['update_step']} updates"
    return "No checkpoints found, run from scratch"

if __name__ == "__main__":
    output_dir = "output"

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, choices=["dirs", "convert"], default="dirs")
    parser.add_argument("--file-name", type=str, required=True)
    args = parser.parse_args()

    task = args.task
    if task == "dirs":
        file_name = args.file_name
        file_args_list, pre_args_list, lines = parse_args_from_file(file_name)
        for i, (file_args, pre_args, line) in enumerate(zip(file_args_list, pre_args_list, lines)):
            loc = convert_argparse_to_hash_path(file_args, accelerate_args = pre_args, output_folder = output_dir)
            status = get_run_status(loc)
            print(f"{i+1}: {loc} - {status}")
    elif task == "convert":
        file_name = args.file_name
        file_args_list, pre_args_list, lines = parse_args_from_file(file_name)
        skip_list = []
        for i, (file_args, pre_args, line) in enumerate(zip(file_args_list, pre_args_list, lines)):
            loc1 = convert_argparse_to_hash_path(file_args, accelerate_args = pre_args, output_folder = output_dir)
            if os.path.exists(loc1):
                loc2 = convert_argparse_to_hash_path(file_args, accelerate_args = pre_args, output_folder = output_dir)
                print(f"mv {loc1} {loc2}")
            else:
                skip_list.append(loc1)
        
        for i, loc in enumerate(skip_list):
            print(f"skipping {i}: {loc}")
    elif task == "lm_eval_script":
        file_name = args.file_name
        file_args_list, pre_args_list, lines = parse_args_from_file(file_name)
        for i, (file_args, pre_args, line) in enumerate(zip(file_args_list, pre_args_list, lines)):
            loc, _ = convert_argparse_to_hash_path(file_args, accelerate_args = pre_args, output_folder = output_dir)
            print(f"{i+1}: {loc}")
            if file_args.benchmark_tasks is not None:
                if isinstance(file_args.benchmark_tasks, str):
                    task_list = file_args.benchmark_tasks.split(",")
                else:
                    task_list = args.benchmark_tasks
            else:
                print("No benchmark tasks specified, skipping evaluation")
                task_list = None
            lm_eval_string = get_lm_eval_string(output_dir = loc, 
                                                tokenizer_path = file_args.tokenizer_path, 
                                                tasks=task_list,
                                                # num_fewshot=eval_config["num_fewshot"],
                                                limit=file_args.limit,
                                                log_samples=file_args.log_samples,
                                                # cache_requests=eval_config["cache_requests"],
                                                # show_config=eval_config["show_config"],
                                                experiment = file_args.experiment_name,
                                                )        
            with open(os.path.join(loc, "lm_eval.sh"), "w") as f:
                f.write(lm_eval_string)
    