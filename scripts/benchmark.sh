export TMPDIR="/cmlscratch/astein0/tmp"

# accelerate launch --num_processes 8 -m lm_eval --model hf \
#     --model_args "pretrained=meta-llama/Llama-3.2-1B,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results" \
#     --num_fewshot 4 \
#     --limit 100
    
# accelerate launch --num_processes 8 -m lm_eval \
#     --model_args "pretrained=output/Llama-3.2-1B-task_SFT-finetuning_mode_full-batch64-extend_default/final_model,tokenizer=tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results" \
#     --num_fewshot 4 \
#     --limit 100 \
#     --log_samples

accelerate launch --num_processes 8 -m lm_eval \
    --model_args "pretrained=output/Llama-3.2-1B-task_SFT-finetuning_mode_full-batch64/final_model,tokenizer=meta-llama/Llama-3.2-1B,parallelize=True" \
    --gen_kwargs "do_sample=True,temperature=0.7,top_p=3" \
    --tasks minerva_math \
    --batch_size auto \
    --output_path "./eval_results" \
    --limit 100 \
    --log_samples \
    --cache_requests true \
    --show_config \
    --use_cache lm-evaluation-harness-cache

# accelerate launch --num_processes 8 -m lm_eval \
#     --model_args "pretrained=output/batch_256_checkpointing/final_model,tokenizer=meta-llama/Llama-3.2-1B,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results" \
#     --num_fewshot 4 \
#     --limit 100

# accelerate launch --num_processes 8 -m lm_eval \
#     --model_args "pretrained=output/batch_512/final_model,tokenizer=meta-llama/Llama-3.2-1B,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results" \
#     --num_fewshot 4 \
#     --limit 32 \
#     --log_samples

    

# accelerate launch --num_processes 8 -m lm_eval \
#     --model_args "pretrained=output/batch_512_extend_random/final_model,old_tokenizer=meta-llama/Llama-3.2-1B,tokenizer=/cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start/new_mergeable_ranks_2000.model,pre_tok_name=empty,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results" \
#     --num_fewshot 4 \
#     --limit 32 