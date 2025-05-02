from typing import List, Dict, Union, Optional, Any
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TensorType, PaddingStrategy
from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices

def get_llama32_instruct_chat_template() -> str:
    template = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>" }}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content']|trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
    return template

def get_llama32_instruct_chat_template_minimal() -> str:
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_message | default("You are a helpful assistant.") }}<|eot_id|>{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>

{% if message.role == "assistant" %}
{% generation %}
{{ message.content }}
{% endgeneration %}
{% else %}
{{ message.content }}
{% endif %}
<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% generation %} {% endgeneration %}<|eot_id|>{% endif %}"""
    return template

def get_llama32_repeat_chat_template() -> str:
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant who repeats the given question and answer exactly.<|eot_id|>{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>

{% if message.role == "assistant" %}
{% generation %}
{{ message.content }}
{% endgeneration %}
{% else %}
{{ message.content }}
{% endif %}
<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% generation %} {% endgeneration %}<|eot_id|>{% endif %}"""
    return template

def apply_chat_template_normal(
    tokenizer: AutoTokenizer,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    return_assistant_tokens_mask: bool = False,
    tokenize: bool = True,
    return_tensors: Optional[Union[str, TensorType]] = None,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_dict: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    add_system_message: bool = True,
) -> Union[str, Dict[str, Any]]:
    """
    Normal chat templating (instruction fine-tuning case).
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    if chat_template is None:
        chat_template = get_llama32_instruct_chat_template()

    compiled_template = _compile_jinja_template(chat_template)

    if isinstance(conversation, (list, tuple)) and isinstance(conversation[0], dict):
        conversations = [conversation]
        is_batched = False
    else:
        conversations = conversation
        is_batched = True

    rendered = []
    all_generation_indices = []

    for chat in conversations:
        # TODO
        if add_system_message:
            DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
            chat = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}] + chat

        if return_assistant_tokens_mask:
            rendered_chat, generation_indices = _render_with_assistant_indices(
                compiled_template=compiled_template,
                messages=chat,
                tools=None,
                documents=None,
                add_generation_prompt=add_generation_prompt,
                **tokenizer.special_tokens_map,
            )
            all_generation_indices.append(generation_indices)
        else:
            rendered_chat = compiled_template.render(
                messages=chat,
                tools=None,
                documents=None,
                add_generation_prompt=add_generation_prompt,
                **tokenizer.special_tokens_map,
            )
        rendered.append(rendered_chat)

    if not is_batched:
        rendered = rendered[0]

    if not tokenize:
        return rendered

    out = tokenizer(
        rendered,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors=return_tensors,
        **tokenizer_kwargs,
    )

    if return_assistant_tokens_mask:
        assistant_masks = []
        input_ids = out["input_ids"] if is_batched or return_tensors else [out["input_ids"]]

        if all(len(spans) == 0 for spans in all_generation_indices):
            assistant_masks = build_loss_mask_from_roles(
                input_ids_list=input_ids,
                tokenized_texts=rendered,
                conversation_batches=conversations,
                tokenizer=tokenizer,
            )
        else:
            for i, gen_spans in enumerate(all_generation_indices):
                current_mask = [0] * len(input_ids[i])
                for start_char, end_char in gen_spans:
                    start_token = out.char_to_token(i, start_char)
                    end_token = out.char_to_token(i, end_char - 1)
                    if start_token is None or end_token is None:
                        continue
                    for token_idx in range(start_token, end_token + 1):
                        current_mask[token_idx] = 1
                assistant_masks.append(current_mask)

        if not is_batched and not return_tensors:
            assistant_masks = assistant_masks[0]

        out["loss_mask"] = assistant_masks

    return out

def convert_to_repeat_chat(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert a chat to a repeat chat.
    """
    assert len(chat) == 2, f"Chat must have exactly 2 messages, got {len(chat)}"
    assert chat[0]["role"] == "user", f"First message must be a user message, got {chat[0]['role']}"
    assert chat[1]["role"] == "assistant", f"Second message must be an assistant message, got {chat[1]['role']}"
    sentence = f"Question: {chat[0]['content']} Answer: {chat[1]['content']}"
    return [{"role": "user", "content": sentence}, {"role": "assistant", "content": sentence}]

def apply_chat_template_repeat(
    base_tokenizer: AutoTokenizer,
    second_tokenizer: AutoTokenizer,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    chat_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    return_tensors: Optional[Union[str, TensorType]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Special repeat-task templating (two-tokenizer case), supports batching.
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    if chat_template is None:
        chat_template = get_llama32_repeat_chat_template()

    compiled_template = _compile_jinja_template(chat_template)

    if isinstance(conversation, (list, tuple)) and isinstance(conversation[0], dict):
        conversations = [conversation]
        is_batched = False
    else:
        conversations = conversation
        is_batched = True

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    loss_mask_list = []
    texts = []

    for chat in conversations:
        chat_repeated = convert_to_repeat_chat(chat)
        rendered_chat, generation_indices = _render_with_assistant_indices(
            compiled_template=compiled_template,
            messages=chat_repeated,
            tools=None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **base_tokenizer.special_tokens_map,
        )

        gen_start, gen_end = generation_indices[0]

        base_text = rendered_chat[:gen_start]
        generated_text = rendered_chat[gen_start:gen_end]
        rest_text = rendered_chat[gen_end:]

        out_base = base_tokenizer(base_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)
        out_generated = second_tokenizer(generated_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)
        out_rest = base_tokenizer(rest_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)

        # Concatenate input_ids and attention_mask
        input_ids = out_base["input_ids"]
        attention_mask = out_base["attention_mask"]
        if hasattr(input_ids, "size") and len(input_ids.shape) == 2:
            # torch or numpy
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
        input_ids = list(input_ids) + list(out_generated["input_ids"][0]) + list(out_rest["input_ids"][0])
        attention_mask = list(attention_mask) + list(out_generated["attention_mask"][0]) + list(out_rest["attention_mask"][0])
        labels = input_ids.copy()
        loss_mask = (
            [0] * out_base["input_ids"].shape[-1] +
            [1] * out_generated["input_ids"].shape[-1] +
            [0] * out_rest["input_ids"].shape[-1]
        )

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        loss_mask_list.append(loss_mask)
        texts.append(rendered_chat)

    if return_tensors == "pt":
        import torch
        input_ids_list = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids_list], batch_first=True, padding_value=base_tokenizer.pad_token_id)
        attention_mask_list = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_mask_list], batch_first=True, padding_value=0)
        labels_list = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels_list], batch_first=True, padding_value=-100)
        loss_mask_list = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in loss_mask_list], batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "loss_mask": loss_mask_list,
        "text": texts,
    }

def build_loss_mask_from_roles(
    input_ids_list,
    tokenized_texts: List[str],
    conversation_batches: List[List[Dict[str, str]]],
    tokenizer: AutoTokenizer,
) -> List[List[int]]:
    """
    Build a loss mask manually, assuming:
    - input_ids_list: tokenized input IDs (after rendering full conversation)
    - tokenized_texts: rendered texts (str) matching input_ids
    - conversation_batches: original conversations (list of messages with roles)
    - tokenizer: the tokenizer used
    Returns:
        loss_masks: list of loss masks (1 = assistant response token, 0 = others)
    """
    assistant_masks = []

    for input_ids, rendered_text, conversation in zip(input_ids_list, tokenized_texts, conversation_batches):
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        mask = [0] * len(tokens)

        rendered_cursor = 0
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]

            # Find the rendered position of this message's content
            idx = rendered_text.find(content, rendered_cursor)
            if idx == -1:
                raise ValueError(f"Could not find message content '{content}' in rendered text!")

            rendered_cursor = idx + len(content)

            if role == "assistant":
                # Token-wise: mark tokens corresponding to this span
                # Decode tokens sequentially and match character spans
                current_char_pos = 0
                for token_idx, token in enumerate(tokens):
                    decoded_piece = tokenizer.convert_tokens_to_string([token])
                    piece_len = len(decoded_piece)

                    if idx <= current_char_pos < idx + len(content):
                        mask[token_idx] = 1
                    current_char_pos += piece_len

        assistant_masks.append(mask)

    return assistant_masks

def visualize_loss_mask(input_ids, loss_mask, tokenizer):
    """
    Visualize the loss-masked parts of a tokenized sample with color highlighting and show token IDs.

    Args:
        input_ids (List[int] or Tensor): tokenized input ids
        loss_mask (List[int]): mask where 1 = assistant tokens to supervise
        tokenizer (AutoTokenizer): tokenizer for decoding
    Returns:
        str: visualization string
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    output = []
    for token, token_id, mask in zip(tokens, input_ids, loss_mask):
        token_text = f"{token}"
        if token_id > 128000:
            token_id_text = f"\033[1;31m({token_id})\033[0m"  # White color otherwise
        else:
            token_id_text = f"\033[1;37m({token_id})\033[0m"  # Red color if ID > 128000
        if mask:
            # Highlight masked tokens (green background + black text)
            output.append(f"\033[1;30;42m {token_text} \033[0m{token_id_text}")
        else:
            output.append(f"{token_text}{token_id_text}")
    return " ".join(output)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load tokenizers
    base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    second_tokenizer = AutoTokenizer.from_pretrained("tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000")

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    if second_tokenizer.pad_token is None:
        second_tokenizer.pad_token = second_tokenizer.eos_token
        
    # Sample conversations
    normal_conversations = [
        [
            {"role": "user", "content": "What's 2 + 2?"},
            {"role": "assistant", "content": "4"}
        ],
        [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken cross the road?"}
        ],
        [
            { 
                "role": "user", 
                "content": "Can you explain the concept of equitable distribution in family law?" 
            }, 
            { 
                "role": "assistant", 
                "content": "In family law, equitable distribution refers to the process of dividing marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The goal of equitable distribution is to allocate the marital estate in a way that is just and equitable, considering the unique circumstances of each case.\n\nHere are some key principles and concepts related to equitable distribution:\n\n1. **Marital Property**: Marital property includes all assets and debts acquired during the marriage, regardless of whose name they are in. This can include real estate, bank accounts, investments, retirement accounts, vehicles, and personal property.\n2. **Separate Property**: Separate property, on the other hand, refers to assets and debts acquired before the marriage, inherited, or gifted to one spouse individually. These assets are typically not subject to equitable distribution.\n3. **Equitable Does Not Mean Equal**: While the term \"equitable\" might suggest a 50/50 split, that's not always the case. The court's goal is to divide the marital estate fairly, considering various factors, which may result in an unequal distribution.\n4. **Factors Considered**: Courts consider several factors when determining an equitable distribution, including:\n\t* The length of the marriage\n\t* The income and earning capacity of each spouse\n\t* The age and health of each spouse\n\t* The contributions of each spouse to the acquisition of marital assets\n\t* The value of each spouse's separate property\n\t* The tax implications of the distribution\n\t* The needs of each spouse, including their financial situation and living arrangements\n5. **Types of Distribution**: There are two main types of distribution:\n\t* **In-Kind Distribution**: Assets are divided in their current form, such as one spouse keeping the family home and the other spouse keeping a retirement account.\n\t* **Monetary Distribution**: Assets are sold or liquidated, and the proceeds are divided between the spouses.\n6. **Debt Distribution**: Marital debts, such as credit card debt, mortgages, and loans, are also subject to equitable distribution. The court will consider which spouse is more responsible for the debt and allocate it accordingly.\n7. **Valuation**: The value of marital assets and debts must be determined to facilitate an equitable distribution. This may involve appraisals, expert testimony, or other methods to establish the value of complex assets, such as businesses or investments.\n8. **Negotiation and Mediation**: Spouses can negotiate an equitable distribution agreement through mediation or collaborative law, which can help avoid costly and time-consuming litigation.\n9. **Court Intervention**: If an agreement cannot be reached, the court will intervene and make a determination on the equitable distribution of the marital estate.\n\nIn summary, equitable distribution is a complex process that aims to divide marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The court considers various factors to ensure a just and equitable outcome, which may not always result in an equal split." 
            }
        ]
    ]

    repeat_conversation = [
        # [
        #     # {"role": "user", "content": "Repeat this question and answer exactly: What is 5+5? 10."},
        # ],
        [
            { 
                "role": "user", 
                "content": "Can you explain the concept of equitable distribution in family law?" 
            }, 
            { 
                "role": "assistant", 
                "content": "In family law, equitable distribution refers to the process of dividing marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The goal of equitable distribution is to allocate the marital estate in a way that is just and equitable, considering the unique circumstances of each case.\n\nHere are some key principles and concepts related to equitable distribution:\n\n1. **Marital Property**: Marital property includes all assets and debts acquired during the marriage, regardless of whose name they are in. This can include real estate, bank accounts, investments, retirement accounts, vehicles, and personal property.\n2. **Separate Property**: Separate property, on the other hand, refers to assets and debts acquired before the marriage, inherited, or gifted to one spouse individually. These assets are typically not subject to equitable distribution.\n3. **Equitable Does Not Mean Equal**: While the term \"equitable\" might suggest a 50/50 split, that's not always the case. The court's goal is to divide the marital estate fairly, considering various factors, which may result in an unequal distribution.\n4. **Factors Considered**: Courts consider several factors when determining an equitable distribution, including:\n\t* The length of the marriage\n\t* The income and earning capacity of each spouse\n\t* The age and health of each spouse\n\t* The contributions of each spouse to the acquisition of marital assets\n\t* The value of each spouse's separate property\n\t* The tax implications of the distribution\n\t* The needs of each spouse, including their financial situation and living arrangements\n5. **Types of Distribution**: There are two main types of distribution:\n\t* **In-Kind Distribution**: Assets are divided in their current form, such as one spouse keeping the family home and the other spouse keeping a retirement account.\n\t* **Monetary Distribution**: Assets are sold or liquidated, and the proceeds are divided between the spouses.\n6. **Debt Distribution**: Marital debts, such as credit card debt, mortgages, and loans, are also subject to equitable distribution. The court will consider which spouse is more responsible for the debt and allocate it accordingly.\n7. **Valuation**: The value of marital assets and debts must be determined to facilitate an equitable distribution. This may involve appraisals, expert testimony, or other methods to establish the value of complex assets, such as businesses or investments.\n8. **Negotiation and Mediation**: Spouses can negotiate an equitable distribution agreement through mediation or collaborative law, which can help avoid costly and time-consuming litigation.\n9. **Court Intervention**: If an agreement cannot be reached, the court will intervene and make a determination on the equitable distribution of the marital estate.\n\nIn summary, equitable distribution is a complex process that aims to divide marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The court considers various factors to ensure a just and equitable outcome, which may not always result in an equal split." 
            }
        ]
    ]

    # --- NORMAL TEMPLATE EXAMPLE ---
    print("\n=== NORMAL INSTRUCTION TEMPLATE ===")
    normal_result = apply_chat_template_normal(
        tokenizer=second_tokenizer,
        conversation=normal_conversations,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )

    print("Keys:", normal_result.keys())
    print("Input IDs shape:", normal_result["input_ids"].shape)
    print("Loss mask shape:", len(normal_result["loss_mask"]))
    print("Sample text (detokenized):")
    print("\n=== NORMAL TASK LOSS MASK VISUALIZATION ===")
    for i in range(len(normal_result["input_ids"])):    
        print(second_tokenizer.batch_decode(normal_result["input_ids"][i], skip_special_tokens=False))
        viz = visualize_loss_mask(
            input_ids=normal_result["input_ids"][i],
            loss_mask=normal_result["loss_mask"][i],
            tokenizer=second_tokenizer,
        )
        print(viz)
        print("\n")

    # --- REPEAT TEMPLATE EXAMPLE ---
    print("\n=== SPECIAL REPEAT TASK TEMPLATE ===")
    repeat_result = apply_chat_template_repeat(
        base_tokenizer=base_tokenizer,
        second_tokenizer=second_tokenizer,
        conversation=repeat_conversation,
        chat_template=get_llama32_repeat_chat_template(),
        add_generation_prompt=False,
        return_tensors="pt",
        # padding="longest",
        # truncation=True,
    )

    print("Keys:", repeat_result.keys())
    print("Input IDs length:", len(repeat_result["input_ids"]))
    print("Loss mask length:", len(repeat_result["loss_mask"]))
    print("Sample text (detokenized):")
    print("\n=== REPEAT TASK LOSS MASK VISUALIZATION ===")
    for i in range(len(repeat_result["input_ids"])):
        print(second_tokenizer.batch_decode(repeat_result["input_ids"][i], skip_special_tokens=False))
        viz = visualize_loss_mask(
            input_ids=repeat_result["input_ids"][i],
            loss_mask=repeat_result["loss_mask"][i],
            tokenizer=second_tokenizer,
        )
        print(viz)
        print("\n")
