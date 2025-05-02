# Test FSDP
# accelerate launch --config_file fsdp_config.yaml finetune.py --dry-run --total-batch-size 64 --max-train-steps 10 --batch-size 1 --eval-steps 5 --learning-rate 1e-5 --checkpointing-steps 1 --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000 --task-name mixed --finetune-params embeddings --embedding-init-strategy merge --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --wandb-tags mixed,embeddings_free,unfreeze1000,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval  --unfreeze-params-steps 1000 --model meta-llama/Llama-3.2-1B-Instruct --fsdp

# lora testing
accelerate launch --num_processes 2 finetune.py --dry-run --total-batch-size 64 --max-train-steps 10 \
    --batch-size 1 --eval-steps 1 --learning-rate 1e-5 --checkpointing-steps 1 \
    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000 \
    --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 5  --finetune-params-after-unfreeze lora --embedding-init-strategy merge \
    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 \
    --wandb-tags mixed,embeddings_free,unfreeze1000,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all \
    --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-1B-Instruct