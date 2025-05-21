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
# (LOGIN WANDB)

# run:
accelerate launch --num_processes 8 finetune.py --run-lm-eval --limit 100 --dry-run --total-batch-size 32 --max-train-steps 1000    --batch-size 8 --eval-batch-size 8 --eval-steps 50  --eval-iters 128 --benchmark-steps 100 --learning-rate 2e-5 --warmup-steps 100 --checkpointing-steps 100    --wandb efficient_tokenization  --dataset magpie-default-tokenized_1000    --task-name mixed    --wandb-tags mixed,baseline,full,new_runs,mbpp   --tokenizer-path  tomg-group-umd/EIM-tokenizer-llama3.2-magpie-100    --num-new-tokens 1000 --experiment-name sf  --benchmark-tasks ifeval  --log-samples --model meta-llama/Llama-3.2-3B --finetune-params full --lr-schedule cosine  --save-results 3
