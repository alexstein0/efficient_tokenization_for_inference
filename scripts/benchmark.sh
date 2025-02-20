# accelerate launch --num_processes 8 -m lm_eval --model hf \
#     --model_args "pretrained=meta-llama/Llama-3.2-1B,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results.json" \
#     --num_fewshot 4 \
#     --limit 100
    
# accelerate launch --num_processes 8 -m lm_eval \
#     --model_args "pretrained=output/batch_128_checkpointing/final_model,tokenizer=meta-llama/Llama-3.2-1B,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
#     --tasks minerva_math \
#     --batch_size 4 \
#     --trust_remote_code \
#     --output_path "./eval_results.json" \
#     --num_fewshot 4 \
#     --limit 100
    
accelerate launch --num_processes 8 -m lm_eval \
    --model_args "pretrained=output/batch_256_checkpointing/final_model,tokenizer=meta-llama/Llama-3.2-1B,parallelize=True,do_sample=True,temperature=0.7,top_p=3" \
    --tasks minerva_math \
    --batch_size 4 \
    --trust_remote_code \
    --output_path "./eval_results.json" \
    --num_fewshot 4 \
    --limit 100