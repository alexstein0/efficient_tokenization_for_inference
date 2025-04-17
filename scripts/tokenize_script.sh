
#with custom tokenizer
# python data_preprocessing.py --dataset-path /fs/cml-projects/llm-pretraining/datasets/raw/genqa/math --save-dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/new_tokenized-10 --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start_10

#with llama tokenizer
# python data_preprocessing.py --dataset-path /fs/cml-projects/llm-pretraining/datasets/raw/genqa/math --save-dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama_tokenized

#translation task
python data_preprocessing.py --dataset-path /fs/cml-projects/llm-pretraining/datasets/raw/genqa/math --save-dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/translation_tokenized-10 --task translation --model meta-llama/Llama-3.2-1B --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start_10