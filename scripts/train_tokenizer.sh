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

python train_tokenizer.py \
    --raw-data-name genqa \
    --ext math \
    --pre-tok-name empty \
    --cont-or-start start \
    --batch-size 1000 \
    --added-tokens 10 \
    --tokenizer-path-old meta-llama/Llama-3.2-1B \
    --tokenizer-source huggingface