# accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=microsoft/phi-4-mini-reasoning     --gen_kwargs do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks=2048     --tasks gsm8k     --batch_size auto     --output_path ./eval_results/phi_gsm8k --log_samples
# accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=microsoft/phi-4-mini-reasoning     \
#     --gen_kwargs do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks=2048     \
#     --tasks gsm8k_cot     --batch_size auto     --output_path ./eval_results/gsm8k \
#     --log_samples  --fewshot_as_multiturn --apply_chat_template

# TEST
# python train_tokenizer.py \
#     --raw-data-name gsm8k_cot_phi_small \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 10 \
#     --added-tokens 100 \
#     --tokenizer-path-old microsoft/phi-4-mini-reasoning \
#     --tokenizer-source huggingface \
#     --save-interval 1,10,50,100 \
#     --dataset-source-path datasets \
#     --save-loc tokenizers/phi_gsm8k_cot_small \
#     --num-proc 10 \
#     --save-tokenized-data


# python train_tokenizer.py \
#     --raw-data-name gsm8k_cot_phi \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 100 \
#     --added-tokens 1000 \
#     --tokenizer-path-old microsoft/phi-4-mini-reasoning \
#     --tokenizer-source huggingface \
#     --save-interval 1,10,50,100,1000 \
#     --dataset-source-path datasets \
#     --save-loc tokenizers/phi_gsm8k_cot \
#     --save-tokenized-data

# python data_preprocessing.py --raw-data-name gsm8k --dataset-path datasets/gsm8k --save-dataset-name tokenized_0 --task default,translation --model microsoft/phi-4-mini-reasoning --tokenizer-path microsoft/phi-4-mini-reasoning --chat-template-name phi
# python data_preprocessing.py --raw-data-name gsm8k --dataset-path datasets/gsm8k --save-dataset-name tokenized_1 --task default,translation --model microsoft/phi-4-mini-reasoning --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-1 --chat-template-name phi
# python data_preprocessing.py --raw-data-name gsm8k --dataset-path datasets/gsm8k --save-dataset-name tokenized_10 --task default,translation --model microsoft/phi-4-mini-reasoning --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-10 --chat-template-name phi
# python data_preprocessing.py --raw-data-name gsm8k --dataset-path datasets/gsm8k --save-dataset-name tokenized_100 --task default,translation --model microsoft/phi-4-mini-reasoning --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-100 --chat-template-name phi
# python data_preprocessing.py --raw-data-name gsm8k --dataset-path datasets/gsm8k --save-dataset-name tokenized_1000 --task default,translation --model microsoft/phi-4-mini-reasoning --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-1000 --chat-template-name phi

# 1000
# accelerate launch --num_processes 2 finetune.py --run-lm-eval --limit 128 --total-batch-size 16 --max-train-steps 1500    --batch-size 2 --eval-batch-size 1 --eval-steps 100 --benchmark-steps 200 --learning-rate 5e-4 --warmup-steps 250   --checkpointing-steps 100    --wandb efficient_tokenization  --dataset gsm8k-default-tokenized_1000,gsm8k-translation-tokenized_1000    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-1000    --wandb-tags mixed,extend_merge,new_runs,gsm8k     --num-new-tokens 1000 --experiment-name gsm8k_phi_baseline_embeddings  --benchmark-tasks gsm8k  --log-samples  --model microsoft/phi-4-mini-reasoning --finetune-params embeddings --lr-schedule cosine --save-results 3 --do-sample --temperature 0.8 --top-p 0.95 --top-k 50

# # 100
# accelerate launch --num_processes 2 finetune.py --run-lm-eval --limit 128 --total-batch-size 16 --max-train-steps 1500    --batch-size 2 --eval-batch-size 1 --eval-steps 100 --benchmark-steps 200 --learning-rate 5e-4 --warmup-steps 250   --checkpointing-steps 100    --wandb efficient_tokenization  --dataset gsm8k-default-tokenized_100,gsm8k-translation-tokenized_100    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-100    --wandb-tags mixed,extend_merge,new_runs,gsm8k     --num-new-tokens 100 --experiment-name gsm8k_phi_baseline_embeddings  --benchmark-tasks gsm8k  --log-samples  --model microsoft/phi-4-mini-reasoning --finetune-params embeddings --lr-schedule cosine --save-results 3 --do-sample --temperature 0.8 --top-p 0.95 --top-k 50

# # 10
accelerate launch --num_processes 2 finetune.py --run-lm-eval --limit 128 --total-batch-size 16 --max-train-steps 1500    --batch-size 2 --eval-batch-size 1 --eval-steps 100 --benchmark-steps 200 --learning-rate 5e-4 --warmup-steps 250   --checkpointing-steps 100    --wandb efficient_tokenization  --dataset gsm8k-default-tokenized_10,gsm8k-translation-tokenized_10    --task-name mixed --embedding-init-strategy merge    --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-10    --wandb-tags mixed,extend_merge,new_runs,gsm8k     --num-new-tokens 10 --experiment-name gsm8k_phi_baseline_embeddings  --benchmark-tasks gsm8k  --log-samples  --model microsoft/phi-4-mini-reasoning --finetune-params embeddings --lr-schedule cosine --save-results 3 --do-sample --temperature 0.8 --top-p 0.95 --top-k 50

# # 0
# accelerate launch --num_processes 2 finetune.py --run-lm-eval --limit 128 --total-batch-size 16 --max-train-steps 1500    --batch-size 2 --eval-batch-size 1 --eval-steps 100 --benchmark-steps 200 --learning-rate 5e-4 --warmup-steps 250   --checkpointing-steps 100    --wandb efficient_tokenization  --dataset gsm8k-default-tokenized_0,gsm8k-translation-tokenized_0    --task-name mixed    --wandb-tags mixed,baseline,new_runs,gsm8k   --tokenizer-path microsoft/phi-4-mini-reasoning    --num-new-tokens 0 --experiment-name gsm8k_phi_baseline_embeddings  --benchmark-tasks gsm8k  --log-samples --model microsoft/phi-4-mini-reasoning --finetune-params embeddings --lr-schedule cosine --save-results 3 --do-sample --temperature 0.8 --top-p 0.95 --top-k 50


# accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=microsoft/phi-4-mini-reasoning     --gen_kwargs do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks=2048     --tasks gsm8k     --batch_size auto     --output_path ./eval_results/gsm8k_phi_baseline_embeddings --log_samples --apply_chat_template
# accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=output/gsm8k_phi_baseline_embeddings/32564b1b-phi-4-mini-reasoning-mixed-0/final_model,tokenizer=microsoft/phi-4-mini-reasoning     --gen_kwargs do_sample=True,temperature=0.8,top_p=0.95,max_gen_toks=2048     --tasks gsm8k     --batch_size auto     --output_path ./eval_results/gsm8k_phi_baseline_embeddings --log_samples --apply_chat_template  --num-fewshot 5
