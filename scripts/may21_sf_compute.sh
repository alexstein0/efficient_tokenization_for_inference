# SF COMPUTE
# buy:
# sf buy -d '1h' -t h100v

# to see:
# sf vm list 

# to monitor it coming up:
# sf vm logs -f

# to ssh:
# sf vm ssh root@<NUM>


# Workflow:
# conda activate session_env
# git clone https://github.com/alexstein0/efficient_tokenization_for_inference.git
# cd efficient_tokenization_for_inference
# pip install -r requirements.txt
# huggingface-cli login
# wandb login

# run:
# accelerate launch --num_processes 8 finetune.py --dry-run \
#     --max-train-steps 1000  --total-batch-size 32 --batch-size 8 --eval-batch-size 8 \
#     --learning-rate 2e-5 --lr-schedule cosine --warmup-steps 100 \
#     --dataset magpie-default-tokenized_1000 --model meta-llama/Llama-3.2-3B --finetune-params full   --embedding-init-strategy merge --task-name mixed \
#     --tokenizer-path  tomg-group-umd/EIM-tokenizer-llama3.2-magpie-100    --num-new-tokens 1000 \
#     --eval-steps 50  --eval-iters 128 \
#     --run-lm-eval --benchmark-steps 100 --limit 100 --benchmark-tasks ifeval  --log-samples --save-results 3 \
#     --experiment-name sf  --checkpointing-steps 100    --wandb efficient_tokenization  --wandb-tags mixed,baseline,full,new_runs,mbpp

accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision bf16 \
    finetune.py --dry-run \
    --max-train-steps 1000  --total-batch-size 32 --batch-size 8 --eval-batch-size 8 \
    --learning-rate 2e-5 --lr-schedule cosine --warmup-steps 100 \
    --dataset  tomg-group-umd/EIM-dataset-Llama32-magpie-default_0 --model meta-llama/Llama-3.2-3B --finetune-params full   --embedding-init-strategy merge --task-name mixed \
    --tokenizer-path  meta-llama/Llama-3.2-3B-Instruct    --num-new-tokens 0 \
    --eval-steps 50  --eval-iters 128 \
    --run-lm-eval --benchmark-steps 100 --limit 100 --benchmark-tasks ifeval  --log-samples --save-results 3 \
    --experiment-name sf  --checkpointing-steps 100    --wandb efficient_tokenization  --wandb-tags mixed,baseline,full,magpie
    