from typing import List, Dict, Any
import os
import sys
from efficient_tokenization.utils import generate_hashed_dir_name, parse_args
from efficient_tokenization.model_utils import calc_batch_size_stuff

def compile_eval_scripts(all_output_paths: List[str], output_bash_file: str, title: str = ""):
    eval_scripts = []
    missing_list = []

    with open(output_bash_file, "a") as bash_file:
        bash_file.write(f"{title}\n")
        for subdir_path in all_output_paths:

        # for subdir in os.listdir(base_dir):
        #     subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                print(f"No {subdir_path} found, skipping")

            else:
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


def parse_args_from_file(file_path : str):
    """Parse arguments from a file containing command lines"""
    print(f"file_path: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    file_args_list = []
    pre_args_list = []
    python_lines = []
    for line in lines:
        # Split the line into arguments
        args_list = line.split()
        
        try:
            script_idx = args_list.index('finetune.py')
            script_args = args_list[script_idx + 1:]
            pre_args = get_pre_python_args(args_list, script_idx) if script_idx > 0 else {}

            # Use the original parse_args function with modified sys.argv
            original_argv = sys.argv
            sys.argv = ['finetune.py'] + script_args
            file_args = parse_args()
            sys.argv = original_argv
            
            file_args_list.append(file_args)
            pre_args_list.append(pre_args)
            python_lines.append(line)

        except ValueError:
            print(f"Warning: 'finetune.py' not found in line: {line}")
            continue

    return file_args_list, pre_args_list, python_lines


def convert_argparse_to_values(args, num_processes: int = 1):
    ##### BATCH SIZE STUFF #####
       ###### BATCH SIZE STUFF ######
    # logger.info(f"Setting batch size and gradient accumulation steps...")
    total_batch_size, batch_size, gradient_accumulation_steps = calc_batch_size_stuff(total_batch_size = args.total_batch_size, 
                                                                                      batch_size = args.batch_size, 
                                                                                      num_processes = num_processes, 
                                                                                      gradient_accumulate_every = args.gradient_accumulate_every
                                                                                      )

    ###### OUTPUT FILE NAMING STUFF ######
    # logger.info(f"Setting output directory with unique hash of params...")
    # ablation params are:
    model_name = args.model.split('/')[-1] # 0. model
    tokenizer_path = args.tokenizer_path # 1. tokenizer_path
    finetune_params = args.finetune_params # 1. finetune_params
    embedding_init_strategy = args.embedding_init_strategy # 2. embedding_init_strategy
    # dry_run = args.dry_run # 3. dry_run
    # total_batch_size = args.total_batch_size # 4. total_batch_size
    learning_rate = args.learning_rate # 5. learning_rate
    task_name = args.task_name # 6. task_name
    main_loss_type = args.main_loss # 7. main_loss
    num_new_tokens = args.num_new_tokens # 8. num_new_tokens
    # prefreeze params
    if args.unfreeze_params_steps is None or args.unfreeze_params_steps <= 0 or args.finetune_params_prefreeze == args.finetune_params:
        # logger.info(f"Unfreezing params after unfreeze step is not set or is the same as finetune params after unfreeze, so we will not unfreeze params")
        finetune_params_prefreeze = None
        reset_optimizer = False
        unfreeze_params_steps = -1
        warmup_steps_prefreeze = -1
        lr_schedule_prefreeze = None
        will_unfreeze_params = False
    else:
        finetune_params_prefreeze = args.finetune_params_prefreeze
        reset_optimizer = args.reset_optimizer
        unfreeze_params_steps = args.unfreeze_params_steps
        warmup_steps_prefreeze = args.warmup_steps_prefreeze if args.warmup_steps_prefreeze > -1 else unfreeze_params_steps // 10
        lr_schedule_prefreeze = args.lr_schedule_prefreeze  # this is not part of logging
        will_unfreeze_params = True

    dataset_str = args.dataset
    seed = args.seed
    warmup_steps = args.warmup_steps
    lr_schedule = args.lr_schedule

    params_dict = {
        "model_name": model_name,
        "task_name": task_name,
        "finetuning_params": finetune_params,
        "total_batch_size": total_batch_size,
        "learning_rate": learning_rate,
        "main_loss_type": main_loss_type,
        "embedding_init_strategy": embedding_init_strategy,
        "num_new_tokens": num_new_tokens,
        "unfreeze_params_steps": unfreeze_params_steps,
        "finetune_params_prefreeze": finetune_params_prefreeze,
        "dataset": dataset_str,
        "tokenizer_path": tokenizer_path,
        "seed": seed,
        "reset_optimizer": reset_optimizer,
        "warmup_steps": warmup_steps,
        "lr_schedule": lr_schedule,
        "warmup_steps_prefreeze": warmup_steps_prefreeze,
        "lr_schedule_prefreeze": lr_schedule_prefreeze,
    }

    lora_configs = {
        "target_modules": args.lora_target_modules,
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout
    }
    if finetune_params_prefreeze == "lora" or finetune_params == "lora":
        params_dict.update(lora_configs)

    return params_dict

def convert_argparse_to_hash_path(args, accelerate_args = {}, output_folder = "output") -> str:
    # TODO make this equal the finetuning code
    num_processes = int(accelerate_args.get("--num_processes", accelerate_args.get("--num-processes", 1)))
    params_dict = convert_argparse_to_values(args, num_processes)

    hashed_file_name = generate_hashed_dir_name(params_dict, dry_run = args.dry_run)
    if args.extra_info is not None:
        hashed_file_name = f"{hashed_file_name}_{args.extra_info}"

    if args.experiment_name is not None:
        output_folder = os.path.join(output_folder, args.experiment_name)

    output_dir = os.path.join(output_folder, hashed_file_name)
    return output_dir

def convert_results_into_new_token_metrics(results, task_names, tokenizer, base_tokenizer) -> Dict[str, Dict[str, Any]]:
    outputs = {}
    for task_name in task_names:
        samples = results["samples"][task_name]
        filter_val = samples[0]["filter"]
        metrics = samples[0]["metrics"]
        metrics = [f"{m},{filter_val}" for m in metrics]

        result_metrics = results["results"][task_names[0]]
        model_token_counts = []
        theoretical_token_counts = []
        old_theoretical_token_counts = []
        new_token_counts = []
        for sample in samples:
            # TODO make sure the indexes are correct
            input_ids = [int(x) for x in sample["input_ids"][0][0].split(",")]
            model_produced_token_count = len(input_ids)
            theoretical_token_count = len(tokenizer.encode(sample["resps"][0][0], truncation=False, padding="longest", add_special_tokens=False))
            old_theoretical_token_count = len(base_tokenizer.encode(sample["resps"][0][0], truncation=False, padding="longest", add_special_tokens=False))
            model_token_counts.append(model_produced_token_count)
            theoretical_token_counts.append(theoretical_token_count)
            old_theoretical_token_counts.append(old_theoretical_token_count)
            # Count number of new tokens in input_ids
            new_token_id_start = len(base_tokenizer)
            num_new_tokens = sum(1 for tid in input_ids if tid >= new_token_id_start)
            new_token_counts.append(num_new_tokens)

        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric] = result_metrics[metric]

        outputs[task_name] = {
            "metrics": metrics_dict,
            "model_token_counts": model_token_counts,
            "theoretical_token_counts": theoretical_token_counts,
            "old_theoretical_token_counts": old_theoretical_token_counts,
            "new_token_counts": new_token_counts,
        }

    return outputs


def add_baseline_lm_eval(file_args_obj, pre_args: Dict):
    model_path = file_args_obj.model
    tasks = file_args_obj.benchmark_tasks.split(",")
    experiment = file_args_obj.experiment_name
    num_processes = int(pre_args.get("--num_processes", pre_args.get("--num-processes", 1)))
    limit = int(file_args_obj.limit)
    log_samples = bool(file_args_obj.log_samples)
    # cache_requests = bool(file_args_obj.cache_requests)
    # show_config = bool(file_args_obj.show_config)
    # num_fewshot = int(file_args_obj.num_fewshot)

    return get_lm_eval_string(output_dir = model_path, 
                              tokenizer_path = None,
                              base_tokenizer_path=file_args_obj.original_model,
                              tasks = tasks,
                              num_processes = num_processes,
                              limit = limit,
                              log_samples = log_samples,
                            #   cache_requests = cache_requests,
                            #   show_config = show_config,
                            #   num_fewshot = num_fewshot,
                              experiment = experiment)



def get_lm_eval_string(output_dir: str, 
                       tokenizer_path: str = None,
                       base_tokenizer_path: str = None,
                       tasks: List[str] = ["minerva_math"],
                       num_processes: int = 8,
                       limit: int = -1,
                       log_samples: bool = False,
                       cache_requests: bool = False,
                       show_config: bool = False,
                       num_fewshot: int = -1,
                       experiment: str = None,
                       ) -> str:
    model_args_str = (f"pretrained={output_dir},tokenizer={tokenizer_path}") if tokenizer_path is not None else f"pretrained={output_dir}"
    extra_config_str = (f"--extra config base_tokenizer={base_tokenizer_path}") if base_tokenizer_path is not None else ""
    return f"""accelerate launch --num_processes {num_processes} lm_eval_new_tokens \
    --model hf \
    --model_args {model_args_str} \
    --gen_kwargs do_sample=False,temperature=0.0,top_p=1.0 \
    --tasks {",".join(tasks)} \
    --batch_size auto \
    --output_path ./eval_results{"" if experiment == None or len(experiment) == 0 else f"/{experiment}"} \
    --apply_chat_template \
    {'--log_samples ' if log_samples else ''} \
    {'--limit ' + str(limit) if limit > 0 else ''} \
    {'--cache_requests true ' if cache_requests else ''} \
    {'--show_config ' if show_config else ''} \
    {'--num_fewshot ' + str(num_fewshot) if num_fewshot > 0 else ''} \
    {extra_config_str}"""