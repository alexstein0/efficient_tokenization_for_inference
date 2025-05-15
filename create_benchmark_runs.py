import os
from efficient_tokenization.benchmarking_utils import compile_eval_scripts, parse_args_from_file, convert_argparse_to_hash_path, add_baseline_lm_eval, convert_argparse_to_values
from typing import List
import argparse
import shutil

def get_all_output_paths(base_dir: str) -> List[str]:
    all_output_paths = []    
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        all_output_paths.append(os.path.join(base_dir, subdir_path))
    return all_output_paths


if __name__ == "__main__":

    # Example usage:
    args = argparse.ArgumentParser()
    args.add_argument("--base_dir", type=str, default="output")
    args.add_argument("--scripts_dir", type=str, default="scripts")
    args.add_argument("--output_bash_file", type=str, default="compiled_eval_scripts.sh")
    args.add_argument("--tasks", type=str, default="minerva_math")
    args.add_argument("--limit", type=int, default=100)
    args.add_argument("--log_samples", type=bool, default=False)
    args.add_argument("--cache_requests", type=bool, default=False)
    args.add_argument("--show_config", type=bool, default=False)
    args.add_argument("--train_run_file", type=str)
    args.add_argument("--include_baseline", type=bool, default=True)
    args.add_argument("--main_process_port", action="store_true")
    args.add_argument("--num_processes", type=int, default=8)
    args = args.parse_args()

    output_bash_file = os.path.join(args.scripts_dir, args.output_bash_file)
    output_folder = args.base_dir

    file_args_list = []
    pre_args_list = []
    lines = []
    source_script_name = ""
    if args.train_run_file is None:
        output_run_list = get_all_output_paths(args.base_dir)
    else:
        # source_script_name = os.path.join(args.scripts_dir, args.train_run_file)
        source_script_name = args.train_run_file
        file_args_list, pre_args_list, lines = parse_args_from_file(source_script_name)
        output_run_list = []
        for i, (file_args, pre_args, line) in enumerate(zip(file_args_list, pre_args_list, lines)):
            loc = convert_argparse_to_hash_path(file_args, accelerate_args = pre_args, output_folder = args.base_dir)
            output_run_list.append(loc)
            experiment_file_name = args.train_run_file.split("/")[-1]
            shutil.copy2(source_script_name, os.path.join(output_folder, experiment_file_name))
    
    missing_list, process_port = compile_eval_scripts(output_run_list, output_bash_file, title = source_script_name, main_process_port = args.main_process_port, num_processes = args.num_processes)
    if args.include_baseline and len(file_args_list) > 0:
        print(f"Adding baseline model")
        baseline_file_args = file_args_list[0]
        baseline_pre_args = pre_args_list[0]
        base_line_string = add_baseline_lm_eval(baseline_file_args, baseline_pre_args)
        print(base_line_string)
        with open(output_bash_file, "a") as f:
            if process_port is not None:
                base_line_string = base_line_string.replace("--num_processes 8", f"--num_processes {args.num_processes} --main_process_port {process_port}")
            f.write(base_line_string + "\n")

    with open(output_bash_file, "a") as f:
        for missing_dir in missing_list:
            print(f"Missing lm_eval.sh in {missing_dir}, creating here")
            # tokenizer_path = "/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000"
            # lm_eval_string = get_lm_eval_string(missing_dir, tokenizer_path, tasks=args.tasks, limit=args.limit, log_samples=args.log_samples, cache_requests=args.cache_requests, show_config=args.show_config)
            # f.write(lm_eval_string + "\n")

    if output_folder is not None:
        shutil.copy2(output_bash_file, os.path.join(output_folder, args.output_bash_file))
