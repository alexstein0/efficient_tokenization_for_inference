# 1. Create dictionary dataset
# # python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 0  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict
# # python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 1  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict
# python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 10  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict
# # python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 50  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict
# python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 100  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict
# python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 500  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict
# python data_preprocessing.py --raw-data-name dictionary --dataset-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict --save-dataset-name tokenized_dict --task dictionary --model meta-llama/Llama-3.2-1B-Instruct --tokenizer-path /cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000 --chat-template-name llama32 --task dictionary --num-added-tokens 1000  --min-words-per-sample 5 --max-words-per-sample 15 --dictionary-total-samples 500000 --dict-ds-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets/llama-3.2-1000_dict


#2.  finetune base model on magpie 

# 3. Get Datasets for the following tasks:
# 1. Open-ended QA / Reasoning
# hellaswag (if using generative version)
# truthfulqa_generation – open-ended answers to tricky questions
# arc_challenge:gen – generation version (not MCQ)
# gsm8k:gen – step-by-step math reasoning (instead of multiple choice)
# math_qa:gen

accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct     \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=.95,top_k=20,max_gen_toks=2048     \
    --tasks gsm8k     --batch_size auto     --output_path ./eval_results/gsm8k \
    --log_samples  --fewshot_as_multiturn --apply_chat_template



# 📖 2. Summarization / Explanation
# samsum – dialogue summarization
# cnn_dailymail – news summarization
# xsum – extreme summarization
# eli5 – long-form answers to simple questions



# 🧠 3. Creative / Story Generation
# writing_prompts – generate creative stories from prompts
# story_cloze (if generative form is used)


# 💻 4. Code Generation
# mbpp:gen – write Python functions from descriptions
# humaneval:gen – implement code to pass unit tests
# code_completion (task-specific variants)
