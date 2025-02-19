# python finetune.py


# accelerate launch --num_processes 8 finetune_old.py --gradient-accumulate-every 8 --max-train-steps 5 --batch-size 2 --eval-steps 1 --learning-rate 8e-5 --output-dir output/batch_128_TEST --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized 

accelerate launch --num_processes 8 finetune_old.py --gradient-accumulate-every 8 --max-train-steps 50000 --batch-size 2 --eval-steps 1000 --learning-rate 8e-5 --output-dir output/batch_128 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized
accelerate launch --num_processes 8 finetune_old.py --gradient-accumulate-every 16 --max-train-steps 50000 --batch-size 2 --eval-steps 1000 --learning-rate 1.6e-4 --output-dir output/batch_256 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized
accelerate launch --num_processes 8 finetune_old.py --gradient-accumulate-every 32 --max-train-steps 50000 --batch-size 2 --eval-steps 1000 --learning-rate 3.2e-4 --output-dir output/batch_512 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized
