# Train a tokenizer on a dataset
python train_tokenizer.py \
    --raw-data-name genqa \
    --ext math \
    --pre-tok-name empty \
    --cont-or-start start \
    --batch-size 1000 \
    --added-tokens 100 \
    --tokenizer-path-old meta-llama/Llama-3.2-1B \
    --tokenizer-source huggingface

python shrink_tokenizer.py \
    --old-tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000 \
    --num-new-tokens-list 1,5,20,50,100,200,300,400,500,600,700,800,900
                        
# tokenize the dataset
python data_preprocessing.py --dataset-path /fs/cml-projects/llm-pretraining/datasets/raw/genqa/math --save-dataset-name tokenized-10 --task translation --model meta-llama/Llama-3.2-1B --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-10

# now finetune the data:
accelerate launch --num_processes 8 finetune.py --max-train-steps 5000 --total-batch-size 64 --batch-size 2 --eval-steps 100 --learning-rate 4e-5 --checkpointing-steps 1000 --wandb efficient_tokenization  --dataset /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama_tokenized --task-name SFT --finetune-params full --wandb-tags SFT,all_params_free,baseline,new_runs

# monitor the training
python get_run_status.py --file-name scripts/mar20.sh --task dirs

# benchmark the run
python create_benchmark_runs.py --train_run_file scripts/mar20.sh

# gather the benchmarked results
python gather_benchmarks.py --experiment_dir results_10