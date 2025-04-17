# unset TMPDIR

# TESTS

# python train_tokenizer.py \
#     --raw-data-name genqa \
#     --ext math \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 1000 \
#     --added-tokens 1

# python train_tokenizer.py \
#     --raw-data-name genqa \
#     --ext math \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 1000 \
#     --added-tokens 1 \
#     --tokenizer-path-old meta-llama/Llama-3.2-1B \
#     --tokenizer-source huggingface

# REAL

# python train_tokenizer.py \
#     --raw-data-name genqa \
#     --ext math \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 1000 \
#     --added-tokens 10 \
#     --tokenizer-path-old meta-llama/Llama-3.2-1B \
#     --tokenizer-source huggingface

# python train_tokenizer.py \
#     --raw-data-name genqa \
#     --ext math \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 1000 \
#     --added-tokens 100 \
#     --tokenizer-path-old meta-llama/Llama-3.2-1B \
#     --tokenizer-source huggingface


# magpie
python train_tokenizer.py \
    --raw-data-name magpie_pro_300k_filtered \
    --dataset-source-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets \
    --tokenized-data-name magpie_tokenized-llama3 \
    --pre-tok-name empty \
    --cont-or-start start \
    --batch-size 1000 \
    --added-tokens 1000 \
    --tokenizer-path-old meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer-source huggingface \
    --save-tokenized-data