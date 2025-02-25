
#with custom tokenizer
python data_preprocessing.py --dataset-path /fs/cml-projects/llm-pretraining/datasets/raw/genqa/math --save-dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized --tokenizer-path /cmlscratch/astein0/LLM-pretraining/LLM-pretraining-tokenization/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start/new_mergeable_ranks_2000.model --pretokenizer-name empty

#with llama tokenizer
python data_preprocessing.py --dataset-path /fs/cml-projects/llm-pretraining/datasets/raw/genqa/math --save-dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized