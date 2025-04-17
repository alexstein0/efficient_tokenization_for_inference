# #test
# accelerate launch --num_processes 8 finetune.py --dry-run --total-batch-size 64 --max-train-steps 10 \
#     --batch-size 2 --eval-batch-size 1 --eval-steps 2 --learning-rate 1e-5 --checkpointing-steps 250 \   
#     --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000 \   
#     --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 5  --finetune-params-after-unfreeze lora --embedding-init-strategy merge \   
#     --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 \   
#     --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all \
#     --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct --eval-iters 10

# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --finetune-params embeddings --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct

# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze lora --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct

# accelerate launch --num_processes 2 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze full --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct

# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze lora --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie_instruct  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct --extra-info linear_names


# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze lora --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie-3B-loratests  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct --extra-info linear_8_16_05 --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze lora --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie-3B-loratests  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct --extra-info linear_4_8_05  --lora-target-modules linear --lora-r 4 --lora-alpha 8  --lora-dropout 0.05

# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie-3B-loratests  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct  --finetune-params lora --extra-info linear_8_16_05 --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie-3B-loratests  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct  --finetune-params lora --extra-info linear_4_8_05  --lora-target-modules linear --lora-r 4 --lora-alpha 8  --lora-dropout 0.05

# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie-3B-loratests  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct  --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze lora --reset-optimizer --extra-info linear_8_16_05 --lora-target-modules linear --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
# accelerate launch --num_processes 8 finetune.py --total-batch-size 64 --max-train-steps 7000    --batch-size 1 --eval-batch-size 1 --eval-steps 100 --learning-rate 1e-5 --checkpointing-steps 250    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000,magpie-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000    --wandb-tags mixed,embeddings_free,unfreeze1000,lora,extend_merge,new_runs,magpie --eval-losses-to-track new_tokens,all    --num-new-tokens 1000 --experiment-name magpie-3B-loratests  --benchmark-tasks ifeval --model meta-llama/Llama-3.2-3B-Instruct  --finetune-params embeddings  --unfreeze-params-steps 500  --finetune-params-after-unfreeze lora --reset-optimizer --extra-info linear_4_8_05  --lora-target-modules linear --lora-r 4 --lora-alpha 8  --lora-dropout 0.05



# benchmarking
# accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct     --gen_kwargs do_sample=False,temperature=0.0,top_p=1     --tasks ifeval     --batch_size auto     --output_path ./eval_results/magpie_instruct
# accelerate launch --num_processes 8 -m lm_eval     \
#     --model hf --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct     \
#     --tasks ifeval --output_path ./eval_results/magpie_instruct




  
accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=output/magpie_instruct/04231619-Llama-3.2-3B-Instruct-mixed-1000/final_model,tokenizer=/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000     --gen_kwargs do_sample=False,temperature=0.0,top_p=1.0     --tasks ifeval     --batch_size auto     --output_path ./eval_results/magpie_instruct
hf (pretrained=output/magpie_instruct/04231619-Llama-3.2-3B-Instruct-mixed-1000/final_model,tokenizer=/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000), 
gen_kwargs: (do_sample=False,temperature=0.0,top_p=1.0), limit: None, num_fewshot: None, batch_size: auto
|Tasks |Version|Filter|n-shot|        Metric         |   |Value |   |Stderr|
|------|------:|------|-----:|-----------------------|---|-----:|---|------|
|ifeval|      4|none  |     0|inst_level_loose_acc   |↑  |0.5995|±  |   N/A|
|      |       |none  |     0|inst_level_strict_acc  |↑  |0.5540|±  |   N/A|
|      |       |none  |     0|prompt_level_loose_acc |↑  |0.4806|±  |0.0215|
|      |       |none  |     0|prompt_level_strict_acc|↑  |0.4251|±  |0.0213|

hf (pretrained=meta-llama/Llama-3.2-3B-Instruct), gen_kwargs: (do_sample=False), limit: None, num_fewshot: None, batch_size: auto
|Tasks |Version|Filter|n-shot|        Metric         |   |Value |   |Stderr|
|------|------:|------|-----:|-----------------------|---|-----:|---|------|
|ifeval|      4|none  |     0|inst_level_loose_acc   |↑  |0.6607|±  |   N/A|
|      |       |none  |     0|inst_level_strict_acc  |↑  |0.5983|±  |   N/A|
|      |       |none  |     0|prompt_level_loose_acc |↑  |0.5250|±  |0.0215|
|      |       |none  |     0|prompt_level_strict_acc|↑  |0.4510|±  |0.0214|