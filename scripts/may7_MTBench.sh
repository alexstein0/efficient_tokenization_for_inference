# python FastChat/fastchat/llm_judge/gen_model_answer.py --model-path /cmlscratch/astein0/efficient_tokenization_for_inference/output/patching/Llama-3.2-3B-Instruct_plus_e08a94c1-Llama-3.2-3B-mixed-1000 \
#     --model-id Llama-3.2-3B-Instruct_1000_patched  --dtype bfloat16 --answer_file /cmlscratch/astein0/efficient_tokenization_for_inference/mt_bench_outputs/Llama-3.2-3B-Instruct_1000_patched.jsonl  \
#     --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000  \
#     --num-gpus-total 2 --num-gpus-per-model 2

python FastChat/fastchat/llm_judge/gen_judgment.py --model-list Llama-3.2-3B-Instruct_1000_patched \
    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 \
    --judge-model gpt-4o-2024-05-13 --parallel 1 --first-n 1

# python FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-4o-2024-05-13
