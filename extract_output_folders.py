from efficient_tokenization.utils import generate_hashed_dir_name, parse_args
from efficient_tokenization.model_utils import calc_batch_size_stuff
import argparse
import sys
from typing import List, Any, Dict

def convert_argparse_to_values(args, accelerate_args = {}):
    model_name = args.model.split('/')[-1] # 0. model
    finetuning_params = args.finetune_params # 1. finetune_params
    embedding_init_strategy = args.embedding_init_strategy # 2. embedding_init_strategy
    dry_run = args.dry_run # 3. dry_run
    # total_batch_size = args.total_batch_size # 4. total_batch_size
    learning_rate = args.learning_rate # 5. learning_rate
    task_name = args.task_name # 6. task_name
    main_loss_type = args.main_loss # 7. main_loss
    num_new_tokens = args.num_new_tokens
    unfreeze_params_steps = args.unfreeze_params_steps

    num_processes = int(accelerate_args.get("--num_processes", accelerate_args.get("--num-processes", 1)))

    total_batch_size, _, _ = calc_batch_size_stuff(total_batch_size = args.total_batch_size, 
                                                   batch_size = args.batch_size, 
                                                   num_processes = num_processes, 
                                                   gradient_accumulate_every = args.gradient_accumulate_every
                                                   )

    params_dict = {
        "model_name": model_name,
        "task_name": task_name,
        "finetuning_params": finetuning_params,
        "total_batch_size": total_batch_size,
        "learning_rate": learning_rate,
        "main_loss_type": main_loss_type,
        "embedding_init_strategy": embedding_init_strategy,
        "num_new_tokens": num_new_tokens,
        "unfreeze_params_steps": unfreeze_params_steps,
    }

    return generate_hashed_dir_name(params_dict)

def get_pre_python_args(args_list: List[str], script_idx: int) -> Dict[str, Any]:
    pre_args = {}
    ind = 0
    while ind < script_idx:
        if args_list[ind][0:2] == "--":
            if ind + 1 < script_idx:
                pre_args[args_list[ind]] = args_list[ind+1]
                ind += 2
            else:
                print(f"unable to parse pre_arg at ind {ind}: {args_list[ind]} (reached end)")
                break
        else:
            pre_args[args_list[ind]] = None
            ind += 1
    return pre_args

def parse_args_from_file(file_path):
    """Parse arguments from a file containing command lines"""
    print(f"file_path: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    all_dirs = []
    python_lines = []
    for line in lines:
        # Split the line into arguments
        args_list = line.split()
        
        try:
            script_idx = args_list.index('finetune.py')
            print(args_list)
            script_args = args_list[script_idx + 1:]
            pre_args = get_pre_python_args(args_list, script_idx) if script_idx > 0 else {}

            # Use the original parse_args function with modified sys.argv
            original_argv = sys.argv
            sys.argv = ['finetune.py'] + script_args
            file_args = parse_args()  # This uses the imported function
            sys.argv = original_argv
            
            output_dir = convert_argparse_to_values(file_args, accelerate_args = pre_args)
            all_dirs.append(output_dir)
            python_lines.append(line)

        except ValueError:
            print(f"Warning: 'finetune.py' not found in line: {line}")
            continue

    return all_dirs, python_lines

if __name__ == "__main__":
    output_dir = "output"

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file_name

    all_dirs, lines = parse_args_from_file(file_name)

    for i, (loc, line) in enumerate(zip(all_dirs, lines)):
        print(i)
        print(f"{line}")
        print(f"\t{loc}")
        print()

    