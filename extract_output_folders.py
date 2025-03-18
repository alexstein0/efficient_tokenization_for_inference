import os
from efficient_tokenization.utils import generate_hashed_dir_name
import argparse
from finetune import parse_args
from copy import deepcopy


def parse_args_from_file(file_path, line_number=None):
    """Parse arguments from a file containing command lines"""
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if line_number is not None:
        if line_number < 1 or line_number > len(lines):
            raise ValueError(f"Line number {line_number} out of range (1-{len(lines)})")
        lines = [lines[line_number - 1]]
    
    all_args = []
    for line in lines:
        # Split the line into arguments
        args_list = line.split()
        
        # Find the index of finetune.py to extract only script arguments
        try:
            script_idx = args_list.index('finetune.py')
            script_args = args_list[script_idx + 1:]
            
            # Use the original parse_args function with modified sys.argv
            import sys
            original_argv = sys.argv
            sys.argv = ['finetune.py'] + script_args
            parsed_args = parse_args()  # This uses the imported function
            sys.argv = original_argv
            
            all_args.append(parsed_args)
        except ValueError:
            print(f"Warning: 'finetune.py' not found in line: {line}")
            continue
    
    return all_args[0] if len(all_args) == 1 else all_args

if __name__ == "__main__":
    output_dir = "output"

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, required=True)
    args = parser.parse_args()

    file_name = args.file_name

    all_args = parse_args_from_file(file_name)

    for args in all_args:
        print(args)

    