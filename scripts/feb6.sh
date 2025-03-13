# export TMPDIR="/cmlscratch/astein0/tmp"

# python finetune.py

# # TESTS
# accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 5 --batch-size 1 --eval-steps 10 --learning-rate 8e-5 --output-dir output/batch_128_TEST_masked_translation --checkpointing-steps 1000 --wandb efficient_tokenization  --wandb-tags test,batch128,translation  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/translation_tokenized --task translation  --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --embedding-init-strategy mean
# accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 5 --batch-size 2 --eval-steps 1 --learning-rate 8e-5 --output-dir output/batch_128_TEST --checkpointing-steps 1000 --wandb efficient_tokenization  --wandb-tags test,batch128  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama_tokenized 
# accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 1 --batch-size 2 --eval-steps 1 --learning-rate 8e-5 --output-dir output/extend_test --checkpointing-steps 1000 --wandb efficient_tokenization  --wandb-tags test,batch128,extend  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --embedding-init-strategy mean

# Baseline
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 4e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama_tokenized --task-name SFT --finetune-params all --wandb-tags SFT,all_params_free,baseline,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 128 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 8e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama_tokenized --task-name SFT --finetune-params all --wandb-tags SFT,all_params_free,baseline,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 256 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 1.6e-4 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama_tokenized --task-name SFT --finetune-params all --wandb-tags SFT,all_params_free,baseline,new_runs --eval-losses-to-track new_tokens,all

# Extend Mean
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 7000 --batch-size 2 --eval-steps 100 --learning-rate 4e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy mean --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_mean,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 128 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 8e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy mean --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_mean,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 256 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 1.6e-4 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy mean --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_mean,new_runs --eval-losses-to-track new_tokens,all

# Extend zeros
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 7000 --batch-size 2 --eval-steps 100 --learning-rate 4e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy zeros --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_zeros,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 128 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 8e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy zeros --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_zeros,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 256 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 1.6e-4 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy zeros --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_zeros,new_runs --eval-losses-to-track new_tokens,all

# Extend merge
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 7000 --batch-size 2 --eval-steps 100 --learning-rate 4e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy merge --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_merge,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 128 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 8e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy merge --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_merge,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 256 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 1.6e-4 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy merge --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_merge,new_runs --eval-losses-to-track new_tokens,all

# Extend default
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 64 --max-train-steps 7000 --batch-size 2 --eval-steps 100 --learning-rate 4e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy default --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_default,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 128 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 8e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy default --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_default,new_runs --eval-losses-to-track new_tokens,all
accelerate launch --num_processes 8 finetune_old.py --total-batch-size 256 --max-train-steps 5000 --batch-size 2 --eval-steps 100 --learning-rate 1.6e-4 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --task-name SFT --finetune-params all --embedding-init-strategy default --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 --wandb-tags SFT,all_params_free,extend_default,new_runs --eval-losses-to-track new_tokens,all
