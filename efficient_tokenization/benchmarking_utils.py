from typing import List
def get_lm_eval_string(output_dir: str, 
                       tokenizer_path: str, 
                       tasks: List[str],
                       num_processes: int = 8,
                       limit: int = -1,
                       log_samples: bool = False,
                       cache_requests: bool = False,
                       show_config: bool = False,
                       num_fewshot: int = -1,
                       ) -> str:
    return f"""accelerate launch --num_processes {num_processes} -m lm_eval \
    --model_args pretrained={output_dir}/final_model,tokenizer={tokenizer_path} \
    --gen_kwargs do_sample=True,temperature=0.7,top_p=3 \
    --tasks {",".join(tasks)} \
    --batch_size auto \
    --output_path ./eval_results \
    {'--log_samples ' if log_samples else ''} \
    {'--limit ' + str(limit) if limit > 0 else ''} \
    {'--cache_requests true ' if cache_requests else ''} \
    {'--show_config ' if show_config else ''} \
    {'--num_fewshot ' + str(num_fewshot) if num_fewshot > 0 else ''}"""