# # just embeddings and first_last
# accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
# accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 

# # just embeddings and lora
# accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
# accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05

# # # just embeddings and lora
# accelerate launch --num_processes 2 --main_process_port 29500 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29501 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29502 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer


# # just embeddings and first_last
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 

# # just embeddings and lora
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05

# # # just embeddings and lora
# accelerate launch --num_processes 2 --main_process_port 29503 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29504 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29505 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_100,magpie-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-100    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 100 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer


# # just embeddings and first_last
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 

# # just embeddings and lora
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05

# # # just embeddings and lora
# accelerate launch --num_processes 2 --main_process_port 29506 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29507 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29508 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_10,magpie-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-10    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 10 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer


# # just embeddings and first_last
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze first_last  --reset-optimizer 

# # just embeddings and lora
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05

# # # just embeddings and lora
# accelerate launch --num_processes 2 --main_process_port 29509 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29510 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29511 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_0,magpie-translation-tokenized_0    --task-name SFT --tokenizer-path meta-llama/Llama-3.2-3B    --wandb-tags SFT,baseline,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 0 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params embeddings   --unfreeze-params-steps 1000  --finetune-params-after-unfreeze full  --reset-optimizer


##### bonus to try just loraing from the beginning:

# # just embeddings and first_last
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params first_last  --reset-optimizer 
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,first_last,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params first_last  --reset-optimizer 

# # just embeddings and lora
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params lora  --reset-optimizer  --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05

# # # just embeddings and lora
# accelerate launch --num_processes 2 --main_process_port 29512 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29513 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 2e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params full  --reset-optimizer
# accelerate launch --num_processes 2 --main_process_port 29514 finetune.py --run-lm-eval --limit 100 --total-batch-size 64 --max-train-steps 5000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 4e-5 --warmup-steps 250 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,full,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name base_model  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B --finetune-params full  --reset-optimizer
