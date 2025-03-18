import os
from efficient_tokenization.benchmarking_utils import get_lm_eval_string

def compile_eval_scripts(base_dir, output_bash_file):
    eval_scripts = []
    missing_list = []

    with open(output_bash_file, "w") as bash_file:
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)

            if os.path.isdir(subdir_path):
                lm_eval_script_path = os.path.join(subdir_path, "lm_eval.sh")

                # Check if lm_eval.sh exists
                if os.path.isfile(lm_eval_script_path):
                    with open(lm_eval_script_path, "r") as f:
                        line = f.read().strip()
                        bash_file.write(line + "\n")
                    eval_scripts.append(subdir_path)
                    print(f"Found lm_eval.sh in {subdir_path}")
                else:
                    # Check if 'final_model' subfolder exists
                    final_model_path = os.path.join(subdir_path, "final_model")
                    if os.path.isdir(final_model_path):
                        missing_list.append(subdir_path)

                    else:
                        print(f"No lm_eval.sh or final_model found in {subdir_path}, maybe still training?")

    print(f"Compiled lm_eval scripts from {len(eval_scripts)} directories into {output_bash_file}")
    return missing_list



if __name__ == "__main__":

    # Example usage:
    base_dir = "output"
    scripts_dir = "scripts"
    tasks = ["minerva_math"]
    limit = 100
    log_samples = False
    cache_requests = False
    show_config = False
    output_bash_file = os.path.join(scripts_dir, "compiled_eval_scripts.sh")
    missing_list = compile_eval_scripts(base_dir, output_bash_file)

    with open(output_bash_file, "a") as f:
        for missing_dir in missing_list:
            print(f"Missing lm_eval.sh in {missing_dir}, creating here")
            tokenizer_path = "/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000"
            lm_eval_string = get_lm_eval_string(missing_dir, tokenizer_path, tasks=tasks, limit=limit, log_samples=log_samples, cache_requests=cache_requests, show_config=show_config)
            f.write(lm_eval_string + "\n")

