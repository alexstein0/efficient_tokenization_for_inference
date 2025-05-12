accelerate launch --num_processes 8 -m lm_eval     --model_args pretrained=Qwen/Qwen3-0.6B     \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=.95,top_k=20,max_gen_toks=2048     \
    --tasks gsm8k     --batch_size auto     --output_path ./e6val_results/gsm8k \
    --log_samples  --fewshot_as_multiturn --apply_chat_template

# python train_tokenizer.py \
#     --raw-data-name magpie_pro_300k_filtered \
#     --dataset-source-path /cmlscratch/astein0/efficient_tokenization_for_inference/datasets \
#     --tokenized-data-name magpie_tokenized-qwen \
#     --pre-tok-name empty \
#     --cont-or-start start \
#     --batch-size 100 \
#     --added-tokens 10000 \
#     --save-interval 1,10,50,100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
#     --tokenizer-path-old Qwen/Qwen3-0.6B \
#     --tokenizer-source huggingface \
#     --save-tokenized-data \
#     --save-loc tokenizers/qwen_magpie