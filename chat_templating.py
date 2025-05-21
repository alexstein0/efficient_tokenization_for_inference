from typing import List, Dict, Union, Optional, Any
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TensorType, PaddingStrategy
from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices
import json
import os
import random

def get_system_prompt(chat_template_name: str) -> str:
    if chat_template_name == "qwen":
        return "You are a helpful assistant."
    elif chat_template_name == "phi":
        return 'Your name is Phi, an AI math expert developed by Microsoft.'
    elif chat_template_name == "llama32":
        return "You are a helpful assistant."
    else:
        raise ValueError(f"Unknown chat template: {chat_template_name}")
    
def get_qwen_chat_template() -> str:
    template = """{%- if tools %}
<|im_start|>system
{%- if messages[0].role == 'system' %}
{{ messages[0].content + '\\n\\n' }}
{%- endif %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{% endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }}}
</tool_call><|im_end|>
{%- else %}
  {%- if messages[0].role == 'system' %}
<|im_start|>system
{{ messages[0].content }}
<|im_end|>
  {%- endif %}
{%- endif %}

{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
  {%- set index = (messages|length - 1) - loop.index0 %}
  {%- if ns.multi_step_tool and message.role == "user"
        and not (message.content.startswith('<tool_response>')
                 and message.content.endswith('</tool_response>')) %}
    {%- set ns.multi_step_tool = false %}
    {%- set ns.last_query_index = index %}
  {%- endif %}
{%- endfor %}

{%- for message in messages %}
  {%- if message.role == "user" or (message.role == "system" and not loop.first) %}
<|im_start|>{{ message.role }}
{{ message.content }}
<|im_end|>

  {%- elif message.role == "assistant" %}
    {%- set content = message.content %}
    {%- set reasoning_content = '' %}
    {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
      {%- set reasoning_content = message.reasoning_content %}
    {%- else %}
      {%- if '</think>' in message.content %}
        {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
        {%- set reasoning_content = message.content.split('</think>')[0].split('<think>')[-1].lstrip('\n') %}
      {%- endif %}
    {%- endif %}

    {%- if loop.index0 > ns.last_query_index %}
      {%- if loop.last or (not loop.last and reasoning_content) %}
<|im_start|>assistant
<think>
{{ reasoning_content }}
</think>

{{ content }}
      {%- else %}
<|im_start|>assistant
{{ content }}
      {%- endif %}
    {%- else %}
<|im_start|>assistant
{{ content }}
    {%- endif %}

    {%- if message.tool_calls %}
      {%- for tool_call in message.tool_calls %}
<tool_call>
{"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }}}
</tool_call>
      {%- endfor %}
    {%- endif %}
<|im_end|>

  {%- elif message.role == "tool" %}
    {%- if loop.first or messages[loop.index0 - 1].role != "tool" %}
<|im_start|>user
    {%- endif %}
<tool_response>
{{ message.content }}
</tool_response>
    {%- if loop.last or messages[loop.index0 + 1].role != "tool" %}
<|im_end|>
    {%- endif %}
  {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
<|im_start|>assistant
  {%- if enable_thinking is defined and enable_thinking is false %}
<think>

</think>

  {%- endif %}
{%- endif %}"""
    return template


def get_qwen_repeat_chat_template() -> str:
    # system should be first but isnt special
    template = """{% for message in messages %}<|im_start|>{{ message.role }}
{% if message.role == "assistant" %}
{% generation %}
{{ message.content }}
{% endgeneration %}
{% else %}
{{ message.content }}
{% endif %}
{% if message.role == "system" %}
'\\n\\n'
{% endif %}
<|im_end|>
{% endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
  {%- if enable_thinking is defined and enable_thinking is false %}
<think>

</think>

  {%- endif %}
{%- endif %}"""
    return template


def get_phi_cot_chat_template() -> str:
    template = """{{ '<|system|>Your name is Phi, an AI math expert developed by Microsoft.' }}
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ message['content'] }}
        {% if 'tools' in message and message['tools'] is not none %}
            {{ '<|tool|>' + message['tools'] + '<|/tool|>' }}
        {% endif %}
    {% endif %}
{% endfor %}
{{ '<|end|>' }}
{% for message in messages %}
    {% if message['role'] != 'system' %}
        {{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|assistant|>' }}
{% else %}
    {{ eos_token }}
{% endif %}"""
    return ''.join(line.strip() for line in template.splitlines())

def get_phi_chat_template_repeat() -> str:
    template = """{{ '<|system|>Your name is Phi, an AI math expert developed by Microsoft.' }}
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ message['content'] + '<|end|>' }}
    {% elif message['role'] == 'user' %}
        {{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}
    {% elif message['role'] == 'assistant' %}
        {% generation %}
            {{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}
        {% endgeneration %}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|assistant|>' }}
{% else %}
    {{ eos_token }}
{% endif %}"""
    return ''.join(line.strip() for line in template.splitlines())

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
{%- endif %}"""
    return ''.join(line.strip() for line in template.splitlines())

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
    return ''.join(line.strip() for line in template.splitlines())

def get_llama32_repeat_chat_template() -> str:
    template = """<|begin_of_text|>
{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>
{% if message.role == "assistant" %}
{% generation %}
{{ message.content }}
{% endgeneration %}
{% else %}
{{ message.content }}
{% endif %}
<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% generation %} {% endgeneration %}<|eot_id|>{% endif %}"""
    return ''.join(line.strip() for line in template.splitlines())

# def get_dictionary_chat_template() -> str:
#     template = """{{ bos_token }}<|system|>{{ system_message }}
# {% for message in messages %}
# <|{{ message['role'] }}|>
# {{ message['content'].strip() }}
# {% endfor %}
# {% if add_generation_prompt %}<|assistant|>
# <|assistant|>
# """

# {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
# {%- if tools is not none %}
#     {{- "Environment: ipython\n" }}
# {%- endif %}
# {{- "Cutting Knowledge Date: December 2023\n" }}
# {{- "Today Date: " + date_string + "\n\n" }}
# {%- if tools is not none and not tools_in_user_message %}
#     {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
# {%- endif %}
# {{- system_message }}
# {{- "<|eot_id|>" }}

def get_llama32_dictionary_chat_template() -> str:
    template = """{{ bos_token }}
    {%- if not date_string is defined %}
        {%- if strftime_now is defined %}
            {%- set date_string = strftime_now(\"%d %b %Y\") %}
        {%- else %}
            {%- set date_string = \"15 May 2025\" %}
        {%- endif %}
    {%- endif %}
{% for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}
    {% if message['role'] == 'system' %}
        {{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n
        {{ message.content }}<|eot_id|>
    {% elif message['role'] == 'user' %}
        {% generation %}
            {{ message.content }}
        {% endgeneration %}
        <|eot_id|>
    {% elif message['role'] == 'assistant' %}
        {% generation %}
            {{ message.content }}<|eot_id|>
        {% endgeneration %}
    {% endif %}\n
{% endfor %}
"""
    return ''.join(line.strip() for line in template.splitlines())

def convert_to_repeat_chat(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert a chat to a repeat chat.
    """
    output = []
    if len(chat) == 3:
        assert chat[0]["role"] == "system", f"First message must be a system message, got {chat[0]['role']}"
        output.append({"role": "system", "content": chat[0]["content"]})
        chat = chat[1:]

    assert len(chat) == 2, f"Chat must have exactly 2 messages, got {len(chat)}"
    assert chat[0]["role"] == "user", f"First message must be a user message, got {chat[0]['role']}"
    assert chat[1]["role"] == "assistant", f"Second message must be an assistant message, got {chat[1]['role']}"
    assert isinstance(chat[0]['content'], List), f"First message must be a List, got {type(chat[0]['content'])}"
    sentence = f"Question: {chat[0]['content']} Answer: {chat[1]['content']}"
    return [{"role": "user", "content": sentence}, {"role": "assistant", "content": sentence}]


def convert_to_dictionary_chat(dictionary_sample: List[Dict[str, Any]], tokenizer: AutoTokenizer, system_message: str = ""):
    """
    Convert a chat to a dictionary task chat.
    The data will be a list of dictionaries, each with the following keys:
    - old_tokens: List[int]
    - new_token: int
    - token_id: int
    - merge_ids: List[int]
    There will be any number of these dictionaries, and the system message will be prepended to the conversation.
    # TODO think about if there should be a delimiter between different words
    """
    first_token_str = []
    second_token_str = []
    first_token_list = []
    second_token_list = []
    first_token_loss_mask = []
    second_token_loss_mask = []
    quote_char = "'"
    comma_char = ","
    quote_id = tokenizer.convert_tokens_to_ids([quote_char])[0]
    comma_id = tokenizer.convert_tokens_to_ids([comma_char])[0]
    for dict_entry in dictionary_sample:
            
        old_token = dict_entry["old_tokens"]
        new_token = dict_entry["new_token"]
        token_id = dict_entry["token_id"]
        merge_ids = dict_entry["merge_ids"]
        merge_len = len(merge_ids)
        new_token_str = f"{quote_char}{new_token}{quote_char}"
        if old_token is not None:
            # TODO hacky way to handle None
            old_token_str = f"{quote_char}{''.join(old_token)}{quote_char}"
        else:
            old_token_str = new_token_str
        assert new_token_str == old_token_str, f"New word must equal old word in string space, got {new_token_str} and {old_token_str}"

        new_token_list = [quote_id, token_id, quote_id]
        old_token_list = [quote_id, *merge_ids, quote_id]
        new_token_mask = [0, 1, 0]
        old_token_mask = [0, *[1] * merge_len, 0]

        randomize_order = True
        if randomize_order and random.random() > 0.5:
            first_token_str.append(new_token_str)
            second_token_str.append(old_token_str)
            first_token_list.append(new_token_list)
            second_token_list.append(old_token_list)
            first_token_loss_mask.append(new_token_mask)
            second_token_loss_mask.append(old_token_mask)
        else:
            first_token_str.append(old_token_str)
            second_token_str.append(new_token_str)
            first_token_list.append(old_token_list)
            second_token_list.append(new_token_list)
            first_token_loss_mask.append(old_token_mask)
            second_token_loss_mask.append(new_token_mask)

    full_first_token_str = comma_char.join(first_token_str)
    full_second_token_str = comma_char.join(second_token_str)
    assert full_first_token_str == full_second_token_str, f"First token string must equal second token string, got {full_first_token_str} and {full_second_token_str}"
    for old, new, old_loss, new_loss in zip(second_token_list[:-1], first_token_list[:-1], first_token_loss_mask[:-1], second_token_loss_mask[:-1]):
        old.append(comma_id)
        new.append(comma_id)
        old_loss.append(0)
        new_loss.append(0)
        # assert old == new, f"Old token list must equal new token list, got {old} and {new}"
    
    output = []
    output.append({"role": "system", "content": system_message})

    user_content = f"Repeat the following:"
    # assistant_content = f"\'{new_token_str[0]}\'"
    output.append({"role": "user", "content": user_content})
    output.append({"role": "assistant", "content": ""})
    return output, full_first_token_str, first_token_list, second_token_list, first_token_loss_mask, second_token_loss_mask
    


def apply_chat_template_normal(
    tokenizer: AutoTokenizer,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    chat_template_name: Optional[str] = None,
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

    if chat_template_name is None:
        chat_template = get_llama32_instruct_chat_template()
    elif chat_template_name == "llama32":
        chat_template = get_llama32_instruct_chat_template()
    elif chat_template_name == "phi":
        chat_template = get_phi_cot_chat_template()
    elif chat_template_name == "qwen":
        chat_template = get_qwen_chat_template()
    else:
        raise ValueError(f"Chat template name {chat_template_name} not found")

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
        # TODO better handle system message
        if add_system_message and "system" not in [m["role"] for m in chat]:
            chat = [{"role": "system", "content": get_system_prompt(chat_template_name)}] + chat

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
    out["text"] = rendered

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


def apply_chat_template_repeat(
    base_tokenizer: AutoTokenizer,
    second_tokenizer: AutoTokenizer,
    conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
    chat_template_name: Optional[str] = None,
    add_generation_prompt: bool = False,
    return_tensors: Optional[Union[str, TensorType]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    add_system_message: bool = True,
) -> Dict[str, Any]:
    """
    Special repeat-task templating (two-tokenizer case), supports batching.
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    if chat_template_name is None:
        chat_template = get_llama32_repeat_chat_template()
    elif chat_template_name == "llama32":
        chat_template = get_llama32_repeat_chat_template()
    elif chat_template_name == "phi":
        chat_template = get_phi_chat_template_repeat()
    elif chat_template_name == "qwen":
        chat_template = get_qwen_repeat_chat_template()
    else:
        raise ValueError(f"Chat template name {chat_template_name} not found")


    compiled_template = _compile_jinja_template(chat_template)
    DEFAULT_REPEAT_SYSTEM_MESSAGE = "You are a helpful assistant who repeats the given question and answer exactly as given but as short as possible"

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
        if add_system_message and "system" not in [m["role"] for m in chat]:
            chat = [{"role": "system", "content": DEFAULT_REPEAT_SYSTEM_MESSAGE}] + chat

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
        else:
            # its a list (THIS IS WHAT IT SHOULD BE)
            input_ids = input_ids + out_generated["input_ids"] + out_rest["input_ids"]
            attention_mask = attention_mask + out_generated["attention_mask"] + out_rest["attention_mask"]
            labels = input_ids.copy()
            loss_mask = (
                [0] * len(out_base["input_ids"]) +
                [1] * len(out_generated["input_ids"]) +
                [0] * len(out_rest["input_ids"])
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

def apply_chat_template_dictionary(
    base_tokenizer: AutoTokenizer,
    second_tokenizer: AutoTokenizer,
    conversation: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    chat_template_name: Optional[str] = None,
    add_generation_prompt: bool = False,
    return_tensors: Optional[Union[str, TensorType]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    add_system_message: bool = True,
) -> Dict[str, Any]:
    """
    Special repeat-task templating (two-tokenizer case), supports batching.
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    # Right now use the repeat template, which is just a generic template for each model.  the important thing is the "convert to dictionary" function
    if chat_template_name is None:
        chat_template = get_llama32_dictionary_chat_template()
    elif chat_template_name == "llama32":
        chat_template = get_llama32_dictionary_chat_template()
    elif chat_template_name == "phi":
        chat_template = get_phi_dictionary_chat_template()
    elif chat_template_name == "qwen":
        chat_template = get_qwen_dictionary_chat_template()
    else:
        raise ValueError(f"Chat template name {chat_template_name} not found")
    

    system_message = get_system_prompt(chat_template_name)
    compiled_template = _compile_jinja_template(chat_template)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    loss_mask_list = []
    texts = []

    for chat in conversation:

        dictionary_chat, raw_string, old_token_ids, new_token_ids, old_token_loss_mask, new_token_loss_mask = convert_to_dictionary_chat(chat, second_tokenizer, system_message)

        rendered_chat, generation_indices = _render_with_assistant_indices(
            compiled_template=compiled_template,
            messages=dictionary_chat,
            tools=None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **base_tokenizer.special_tokens_map,
        )

        assert len(generation_indices) == 2, f"Generation indices must be a list length 2, got {generation_indices}"

        first_gen, first_gen_end = generation_indices[0]
        second_gen, second_gen_end = generation_indices[1]

        # if len(system_message) == 0:
        #     assert first_gen == first_gen_end, f"Generation indices must be the same"
            
        # assert second_gen == second_gen_end, f"Generation indices must be the same"
        
        beginning_text = rendered_chat[:first_gen]
        first_generation_text = rendered_chat[first_gen:first_gen_end]
        between_text = rendered_chat[first_gen_end:second_gen]
        second_generation_text = rendered_chat[second_gen:second_gen_end]
        end_text = rendered_chat[second_gen_end:]

        out_beginning = base_tokenizer(beginning_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)
        out_first_generation = base_tokenizer(first_generation_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)
        out_between = base_tokenizer(between_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)
        out_second_generation = base_tokenizer(second_generation_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)
        out_end = base_tokenizer(end_text, add_special_tokens=False, return_tensors=return_tensors, **tokenizer_kwargs)

        # if randomize_order:
        first_token_ids = old_token_ids
        second_token_ids = new_token_ids
        first_token_loss_mask = old_token_loss_mask
        second_token_loss_mask = new_token_loss_mask

        # Concatenate input_ids and attention_mask
        full_rendered_text = beginning_text + first_generation_text + raw_string + between_text + raw_string + second_generation_text + end_text
        input_ids = (
            out_beginning["input_ids"] + 
            out_first_generation["input_ids"] +
            [x for sublist in first_token_ids for x in sublist] + 
            out_between["input_ids"] +
            [x for sublist in second_token_ids for x in sublist] + 
            out_second_generation["input_ids"] +
            out_end["input_ids"]
            )
        attention_mask = (
            out_beginning["attention_mask"] + 
            out_first_generation["attention_mask"] +
            [1 for sublist in first_token_ids for _ in sublist] + 
            out_between["attention_mask"] + 
            [1 for sublist in second_token_ids for _ in sublist] + 
            out_second_generation["attention_mask"] +
            out_end["attention_mask"]
            )
        labels = input_ids.copy()
        track_old_words = False
        loss_mask = (
            [0] * len(out_beginning["input_ids"]) +
            [0] * len(out_first_generation["input_ids"]) +
            [(x if track_old_words else 0) for sublist in first_token_loss_mask for x in sublist] +
            [0] * len(out_between["input_ids"]) +
            [x for sublist in second_token_loss_mask for x in sublist] +
            [1] * len(out_second_generation["input_ids"]) +
            [0] * len(out_end["input_ids"])
        )
        assert len(loss_mask) == len(input_ids), f"Loss mask must be the same length as input ids, got {len(loss_mask)} and {len(input_ids)}"
        assert len(attention_mask) == len(input_ids), f"Attention mask must be the same length as input ids, got {len(attention_mask)} and {len(input_ids)}"
        assert len(labels) == len(input_ids), f"Labels must be the same length as input ids, got {len(labels)} and {len(input_ids)}"

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        loss_mask_list.append(loss_mask)
        texts.append(full_rendered_text)

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


def visualize_loss_mask(input_ids, tokenizer, loss_mask=None, max_chars_per_line=200, new_token_id_threshold=128000):
    """
    Visualize tokens and their token IDs in a paragraph style,
    with token IDs printed directly underneath, aligned per token by column.
    Handles ANSI color codes for correct alignment.
    """
    import re

    def strip_ansi(s):
        # Remove ANSI escape sequences for accurate visible length
        return re.sub(r'\x1b\[[0-9;]*m', '', s)

    if loss_mask is None:
        loss_mask = [1] * len(input_ids)

    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)

    token_id_pairs = []
    for token, token_id, mask in zip(tokens, input_ids, loss_mask):
        token_display = token
        if mask:
            token_display = f"\033[1;30;42m{token}\033[0m"

        if token_id > new_token_id_threshold:
            id_display = f"\033[1;31m{token_id}\033[0m"
        else:
            id_display = f"\033[1;37m{token_id}\033[0m"

        visible_token_len = len(strip_ansi(token_display))
        visible_id_len = len(strip_ansi(id_display))
        col_width = max(visible_token_len, visible_id_len)

        # Pad token and id manually accounting for ANSI
        padded_token = token_display + ' ' * (col_width - visible_token_len)
        padded_id = id_display + ' ' * (col_width - visible_id_len)

        token_id_pairs.append((padded_token, padded_id, col_width))

    lines = []
    current_token_line = []
    current_id_line = []
    current_width = 0

    for token_fmt, id_fmt, col_width in token_id_pairs:
        if current_width + col_width + 1 > max_chars_per_line and current_token_line:
            lines.append(" ".join(current_token_line))
            lines.append(" ".join(current_id_line))
            lines.append("")
            current_token_line = []
            current_id_line = []
            current_width = 0

        current_token_line.append(token_fmt)
        current_id_line.append(id_fmt)
        current_width += col_width + 1  # +1 for space

    if current_token_line:
        lines.append(" ".join(current_token_line))
        lines.append(" ".join(current_id_line))

    return "\n".join(lines)


def get_example_conversation(
    base_tokenizer: AutoTokenizer,
    second_tokenizer: AutoTokenizer,
):
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

    # repeat_conversation = [
    #     [
    #         { 
    #             "role": "user", 
    #             "content": "Can you explain the concept of equitable distribution in family law?" 
    #         }, 
    #         { 
    #             "role": "assistant", 
    #             "content": "In family law, equitable distribution refers to the process of dividing marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The goal of equitable distribution is to allocate the marital estate in a way that is just and equitable, considering the unique circumstances of each case.\n\nHere are some key principles and concepts related to equitable distribution:\n\n1. **Marital Property**: Marital property includes all assets and debts acquired during the marriage, regardless of whose name they are in. This can include real estate, bank accounts, investments, retirement accounts, vehicles, and personal property.\n2. **Separate Property**: Separate property, on the other hand, refers to assets and debts acquired before the marriage, inherited, or gifted to one spouse individually. These assets are typically not subject to equitable distribution.\n3. **Equitable Does Not Mean Equal**: While the term \"equitable\" might suggest a 50/50 split, that's not always the case. The court's goal is to divide the marital estate fairly, considering various factors, which may result in an unequal distribution.\n4. **Factors Considered**: Courts consider several factors when determining an equitable distribution, including:\n\t* The length of the marriage\n\t* The income and earning capacity of each spouse\n\t* The age and health of each spouse\n\t* The contributions of each spouse to the acquisition of marital assets\n\t* The value of each spouse's separate property\n\t* The tax implications of the distribution\n\t* The needs of each spouse, including their financial situation and living arrangements\n5. **Types of Distribution**: There are two main types of distribution:\n\t* **In-Kind Distribution**: Assets are divided in their current form, such as one spouse keeping the family home and the other spouse keeping a retirement account.\n\t* **Monetary Distribution**: Assets are sold or liquidated, and the proceeds are divided between the spouses.\n6. **Debt Distribution**: Marital debts, such as credit card debt, mortgages, and loans, are also subject to equitable distribution. The court will consider which spouse is more responsible for the debt and allocate it accordingly.\n7. **Valuation**: The value of marital assets and debts must be determined to facilitate an equitable distribution. This may involve appraisals, expert testimony, or other methods to establish the value of complex assets, such as businesses or investments.\n8. **Negotiation and Mediation**: Spouses can negotiate an equitable distribution agreement through mediation or collaborative law, which can help avoid costly and time-consuming litigation.\n9. **Court Intervention**: If an agreement cannot be reached, the court will intervene and make a determination on the equitable distribution of the marital estate.\n\nIn summary, equitable distribution is a complex process that aims to divide marital assets and debts fairly and reasonably between spouses in the event of a divorce or separation. The court considers various factors to ensure a just and equitable outcome, which may not always result in an equal split." 
    #         }
    #     ]
    # ]

    # chat_template = get_qwen_chat_template()
    # chat_template_repeat = get_qwen_repeat_chat_template()
    chat_template_name = "qwen"

    # --- NORMAL TEMPLATE EXAMPLE ---
    # print("\n=== NORMAL INSTRUCTION TEMPLATE ===")
    normal_result = apply_chat_template_normal(
        tokenizer=second_tokenizer,
        conversation=normal_conversations,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        # return_tensors="pt",
        # padding="longest",
        truncation=True,
        chat_template_name=chat_template_name,
    )
    
    # --- REPEAT TEMPLATE EXAMPLE ---
    # print("\n=== SPECIAL REPEAT TASK TEMPLATE ===")
    repeat_result = apply_chat_template_repeat(
        base_tokenizer=base_tokenizer,
        second_tokenizer=second_tokenizer,
        # conversation=repeat_conversation,
        conversation=normal_conversations,
        chat_template_name=chat_template_name,
        add_generation_prompt=False,
        # return_tensors="pt",
        # padding="longest",
        # truncation=True,
    )
    return normal_result, repeat_result

def optimally_tokenize(input_ids: List[int], tokenizer):
    input_ids_as_string = tokenizer.decode(input_ids, skip_special_tokens=False)
    input_ids = tokenizer.encode(input_ids_as_string, add_special_tokens=False)
    return input_ids
    

def ansi_to_html(text: str) -> str:
    """
    Convert ANSI color codes to HTML span tags with inline CSS.
    """
    import re

    ansi_escape = re.compile(r'\x1b\[([0-9;]+)m')

    # Define styles for ANSI codes
    code_to_style = {
        '0': '',  # Reset
        # '1;30;42': 'background-color: #90ee90; color: black;',  # bright green bg
        '1;31': 'color: red;',  # red
        '1;37': 'color: lightgray;',  # gray
        '1;30': 'color: black;',  # fallback if needed
    }

    output = []
    last_end = 0
    open_span = False

    for match in ansi_escape.finditer(text):
        start, end = match.span()
        code = match.group(1)
        style = code_to_style.get(code, '')

        # Append plain text before the match
        output.append(text[last_end:start])

        if open_span:
            output.append('</span>')
            open_span = False

        if style:
            output.append(f'<span style="{style}">')
            open_span = True

        last_end = end

    # Append remaining text
    output.append(text[last_end:])
    if open_span:
        output.append('</span>')

    return ''.join(output).replace('\n', '<br>\n')


def save_visualization_to_html(input_ids, tokenizer, loss_mask=None, html_path="visualization.html", new_token_id_threshold=128000):
    """
    Save the token visualization as an HTML file with color highlighting.
    """
    viz = visualize_loss_mask(
        input_ids=input_ids,
        tokenizer=tokenizer,
        loss_mask=loss_mask,
        new_token_id_threshold=new_token_id_threshold
    )
    html_lines = ['<html><head><style>',
                  'body { font-family: monospace; white-space: pre-wrap; background-color: #1e1e1e; color: #d4d4d4; }',
                  '</style></head><body>']
    html_lines.append(ansi_to_html(viz))
    html_lines.append('</body></html>')

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))


# Save multiple visualizations to a single HTML file, preserving coloring
def save_multiple_visualizations_to_html(viz_strings: Dict[str, str], html_path: str = "visualizations.html"):
    """
    Save multiple visualize_loss_mask() outputs to a single HTML file,
    preserving coloring via ANSI-to-HTML conversion.
    """
    html_lines = ['<html><head><meta charset="utf-8"><style>',
                  'body { font-family: monospace; white-space: pre-wrap; background-color: #1e1e1e; color: #d4d4d4; }',
                  '</style></head><body>']
    for i, (title, viz) in enumerate(viz_strings.items()):
        html_lines.append(f"<h3>{title}</h3>")
        html_lines.append(ansi_to_html(viz))
        html_lines.append("<hr>")
    html_lines.append('</body></html>')

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))


if __name__ == "__main__":

    # Load tokenizers
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    new_tokenizer_name = "tokenizers/Llama-3.2-tokenizer-magpie_pro_300k_filtered-math-empty-start-1000"

    # model_name = "microsoft/phi-4-mini-reasoning"
    # new_tokenizer_name = "/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/phi_gsm8k_cot-1000"

    # model_name = "Qwen/Qwen3-0.6B"
    # new_tokenizer_name = "/cmlscratch/astein0/efficient_tokenization_for_inference/tokenizers/qwen_gsm8k-1000"

    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    second_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name)


    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    if second_tokenizer.pad_token is None:
        second_tokenizer.pad_token = second_tokenizer.eos_token
        

    new_token_id_threshold = len(base_tokenizer)

    def get_step_number(filename):
        try:
            # Extract the number after "_step" and before ".json"
            step_str = filename.split("_step")[1].split(".json")[0]
            return int(step_str)
        except (IndexError, ValueError):
            return float('inf')  # Put files without step numbers at the end

    # can be 'example', 'hardcode', 'directory', or 'file
    
    normal_result = None
    repeat_result = None

    source = "sample_file"

    sample_names = []
    input_id_examples = []
    prompt_list = []
    scores = []
    sample_no = 0
    question_tokenized = None
    question_tokenized_old = None

    if source == "example":
        # user examples
        normal_result, repeat_result = get_example_conversation(
            base_tokenizer=base_tokenizer,
            second_tokenizer=second_tokenizer,
        )
        # dummy stuff bc file is mostly catered to read file or directory
        for i in normal_result:
            sample_names.append('')
            scores.append(0.0)
            prompt_list.append('')

    elif source == "hardcode":
        repeat_result = None

        input_id_examples = [
            [
                        31765, 510, 10931, 596, 836, 50188, 40, 3987, 499, 527, 3815, 1664, 323, 19994, 6220, 2391, 1521, 83321, 3115, 128406, 264, 12637, 19504, 315, 510, 11389, 60, 323, 4423, 25429, 922, 16043, 128385, 4934, 128484, 264, 4545, 311, 2610, 499, 311, 5249, 757, 304, 51582, 701, 26623, 1314, 323, 16043, 128265, 6355, 129038, 510, 4302, 6355, 2457, 948, 3296, 16043, 128313, 527, 51582, 701, 7899, 323, 29820, 128273, 26623, 1920, 128271, 128278, 16188, 13189, 128692, 8396, 382, 53, 11780, 128432, 1193, 3062, 128929, 374, 1101, 264, 44973, 12014, 430, 584, 1288, 682, 1935, 14243, 128406, 264, 26623, 7140, 128388, 17631, 389, 279, 22498, 690, 315, 10495, 311, 6211, 279, 3388, 128692, 3224, 596, 3938, 13, 4718, 7055, 13146, 128257, 16043, 128278, 26632, 430, 584, 1288, 539, 1935, 34504, 382, 2746, 499, 9167, 956, 2736, 11, 1304, 2771, 311, 4254, 311, 7055, 1603, 510, 4302, 12506, 22143, 948, 25532, 7739, 527, 6118, 2561, 2930, 477, 520, 510, 4302, 12506, 3813, 948, 1472, 128547, 1817, 128397, 1614, 596, 6355, 3997, 369, 810, 2038, 389, 12506, 16346, 323, 58982, 382, 1966, 6355, 1938, 11, 1304, 2771, 311, 7055, 4216, 477, 7055, 304, 1732, 128373, 24073, 31744, 17789, 13, 40224, 264, 2764, 3109, 12, 59326, 3110, 323, 387, 2771, 311, 1833, 20562, 128850, 32885, 128262, 12512, 264, 7056, 323, 20958, 3674, 73669, 382, 29690, 11, 16043, 128278, 4443, 5597, 128257, 16043, 369, 4423, 477, 4717, 374, 459, 3927, 5873, 13, 12040, 279, 128686, 39170, 6261, 389, 279, 11426, 11, 4819, 128257, 10396, 430, 5030, 128273, 1455, 311, 499, 128257, 7055, 128397, 42563, 382, 2746, 499, 128395, 3137, 1555, 904, 10742, 477, 90034, 922, 16043, 11, 128421, 39666, 129172, 704, 311, 757, 477, 4423, 775, 499, 7095, 13, 1226, 527, 682, 304, 420, 3871, 128257, 16043, 128278, 22498, 5149, 430, 584, 649, 11322, 3871, 382, 13359, 499, 128494, 128629, 18361, 304, 16043, 129127, 6355, 13, 4718, 7055, 690, 1304, 264, 6811, 128257, 3871, 128579, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 10368, 1057, 26623, 3268, 323, 990, 7119, 128474, 1120, 323, 77109, 8396, 382, 14809, 24886, 3638, 58, 7927, 4076, 60, 510, 7927, 11106, 477, 33907, 311, 11848, 60, 510, 14099, 3729, 2038, 477, 128353, 13777, 422, 8581, 60, 510, 14099, 904, 9959, 12204, 811, 477, 12204, 811, 430, 128414, 9959, 311, 16043, 477, 44973, 20392, 2595, 47, 815, 13, 128674, 11196, 422, 499, 128421, 1440, 1405, 311, 1212, 477, 1148, 311, 1427, 369, 128417, 7055, 13, 50942, 704, 311, 757, 128966, 3358, 8641, 499, 128418, 1920, 323, 1520, 499, 1505, 279, 2038, 499, 1205, 13, 6914, 596, 990, 3871, 128376, 128921, 129091, 6355, 129038, 510, 4302, 6355, 2457, 948, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369, 510, 11389, 948, 6914, 596, 656, 420, 0, 6914, 596, 7055, 0, 6914, 596, 2349, 279, 3388, 128692, 3224, 596, 3938, 13, 6914, 596, 10368, 1057, 26623, 3268, 323, 6211, 264, 53657, 3938, 369

            ],
            [
                        3923, 264, 28254, 323, 6485, 128654, 6104, 1070, 374, 912, 832, 7321, 2269, 1220, 23148, 4320, 128385, 3358, 3493, 499, 449, 1403, 2204, 39555, 128321, 5454, 128800, 19758, 35975, 57277, 8139, 527, 17693, 4443, 323, 649, 3412, 5199, 7438, 369, 7931, 13, 46982, 129111, 1274, 617, 2728, 5370, 5144, 311, 4359, 128698, 791, 10425, 128435, 23392, 128631, 6719, 128435, 1548, 22363, 128631, 36, 385, 40617, 128435, 1548, 22363, 128631, 36, 385, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435, 1548, 22363, 128631, 36, 398, 263, 1494, 128435

            ],
            [
                        30287, 12669, 14767, 11, 4605, 315, 27852, 14559, 128518, 37618, 35907, 271, 30287, 12669, 14767, 11, 1101, 128640, 432, 2706, 13807, 11, 128371, 35482, 1543, 505, 279, 26135, 315, 26523, 889, 16392, 311, 71199, 128722, 51203, 3536, 11639, 13, 38916, 2212, 220, 8190, 20, 11, 432, 2706, 13807, 596, 7126, 11, 3263, 992, 67, 11, 128371, 29947, 41564, 315, 6342, 56018, 8105, 315, 26523, 11, 323, 813, 6691, 11, 30505, 11, 128371, 4562, 128258, 35482, 3838, 315, 358, 9978, 258, 13, 60780, 709, 128265, 26135, 596, 6864, 11, 95411, 11, 432, 2706, 13807, 4036, 264, 1664, 12, 37838, 323, 8040, 128752, 51552, 369, 6411, 26411, 323, 8446, 382, 16397, 279, 88090, 315, 95411, 128709, 8011, 19, 11, 3263, 992, 67, 323, 30505, 596, 86166, 323, 86166, 14948, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018, 8105, 311, 9732, 3263, 992, 67, 439, 38031, 315, 95411, 11, 41577, 279, 129074, 46616, 323, 2109, 6427, 279, 83631, 8603, 382, 16397, 3263, 992, 67, 596, 40061, 11, 95411, 6244, 264, 1401, 6411, 96678, 11, 5737, 129055, 3560, 128265, 9232, 128258, 26135, 596, 24743, 13, 3263, 992, 67, 596, 86166, 323, 19092, 35201, 6197, 56018

            ]
        ]

    elif source == "directory" or source == "file" or source == "sample_file":
        directory = "eval_results/gsm8k_phi_baseline_embeddings_testing/dryrun-a3b8a677-phi-4-mini-reasoning-mixed-1000/"
        # directory = "/cmlscratch/astein0/efficient_tokenization_for_inference/eval_results/gsm8k_phi_baseline_embeddings/a5ad4bba-phi-4-mini-reasoning-mixed-1000"
        if source == "directory":
            for dir_path, dirnames, filenames in os.walk(directory):
                filenames.sort(key=get_step_number)
                for file_name in filenames:
                    with open(os.path.join(dir_path, file_name), "r") as f:
                        result = json.load(f)
                        if sample_no >= 0:
                            # only get one sample here
                            result = [result[sample_no]]
                            prompt = result[0]["arguments"][0][0]
                            if question_tokenized is None:
                                question_tokenized = second_tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                                question_tokenized_old = base_tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

                        sample_in_file = sample_no
                        for sample in result:
                            sample_name = f"{file_name}_{sample_in_file}"
                            print(sample_name)
                            prompt = sample["arguments"][0][0]
                            sample_ids = sample["input_ids"][0][0]

                            score = sample.get("exact_match", -1)
                            scores.append(score)

                            id_strs = sample_ids.split(",")
                            id_ints = [int(x.strip()) for x in id_strs]
                            
                            sample_names.append(sample_name)
                            prompt_list.append(prompt)
                            input_id_examples.append(id_ints)
                            sample_in_file += 1
        elif source == "file":
            # from file examples
            # file_name = "/cmlscratch/astein0/efficient_tokenization_for_inference/eval_results/baseline_embeddings/838db6d0-Llama-3.2-3B-mixed-0/results_stepfinal.json"
            file_name = "/cmlscratch/astein0/efficient_tokenization_for_inference/eval_results/baseline_embeddings/e08a94c1-Llama-3.2-3B-mixed-1000/results_stepfinal.json"
            with open(file_name, "r") as f:
                result = json.load(f)
                # result = [json.loads(line) for line in f]
                for sample in result:
                    prompt = sample["arguments"][0][0]
                    sample_ids = sample["input_ids"][0][0]
                    id_strs = sample_ids.split(",")
                    id_ints = [int(x.strip()) for x in id_strs]

                    sample_names.append(f"{file_name}_{sample_no}")
                    prompt_list.append(prompt)
                    input_id_examples.append(id_ints)
        else:
            # from sample file
            # file_name = "/cmlscratch/astein0/efficient_tokenization_for_inference/eval_results/full_patching/output__full_patching__80c6810f-Llama-3.2-3B-Instruct-mixed-1000__final_model/samples_ifeval_2025-05-14T00-34-25.668535.jsonl"
            file_name = "/cmlscratch/astein0/efficient_tokenization_for_inference/eval_results/full_patching/output__full_patching__ba7c8b60-Llama-3.2-3B-mixed-1000__final_model/samples_ifeval_2025-05-14T01-09-12.508820.jsonl"
            with open(file_name, "r") as f:
                # result = json.load(f)
                result = [json.loads(line) for line in f]
                sample_no = 0
                for sample in result:
                    prompt = sample["arguments"]["gen_args_0"]["arg_0"]
                    sample_ids = sample["input_ids"][0][0]
                    id_strs = sample_ids.split(",")
                    id_ints = [int(x.strip()) for x in id_strs]

                    sample_names.append(f"{file_name}_{sample_no}")
                    prompt_list.append(prompt)
                    input_id_examples.append(id_ints)
                    scores.append("")
                    sample_no += 1

    if normal_result is None:
        # copied examples
        normal_result = {
        "input_ids": input_id_examples,
        "tokenizer": second_tokenizer
        }
        normal_result["loss_mask"] = [None] * len(normal_result["input_ids"])
    
    viz_strings_dict = {}
    viz_strings = []
    viz_strings.append(f"Keys: {normal_result.keys()}")
    viz_strings.append(f"Input IDs shape: {len(normal_result['input_ids'])}")
    viz_strings.append(f"Loss mask shape: {len(normal_result.get('loss_mask', []))}")
    viz_strings.append("Sample text (detokenized):")
    viz_strings.append("\n=== NORMAL TASK LOSS MASK VISUALIZATION ===")
    if question_tokenized is not None:
        viz_strings.append("Question (new tokenizer)")
        viz = visualize_loss_mask(
            input_ids=question_tokenized[0],
            tokenizer=second_tokenizer,
            loss_mask=None,
            new_token_id_threshold=new_token_id_threshold
        )
        viz_strings.append(viz)
        viz_strings_dict["Question (new tokenizer)"] = viz
        viz_strings.append("\n")
        viz_strings.append("Question (old tokenizer)")
        viz = visualize_loss_mask(
            input_ids=question_tokenized_old[0],
            tokenizer=base_tokenizer,
            loss_mask=None,
            new_token_id_threshold=new_token_id_threshold
        )
        viz_strings.append(viz)
        print(question_tokenized_old[0])
        viz_strings_dict["Question (old tokenizer)"] = viz
        viz_strings.append("\n")
    num_examples = len(normal_result["input_ids"])
    for i in range(num_examples):
        viz_strings.append(f"sample_name: {sample_names[i]}, score: {scores[i]}")
        # viz_strings.append(prompt_list[i])
        # viz_strings.append(second_tokenizer.batch_decode(normal_result["input_ids"][i], skip_special_tokens=False))
        viz_strings.append(f"input_length: {len(normal_result['input_ids'][i])}")
        viz = visualize_loss_mask(
            input_ids=normal_result["input_ids"][i],
            tokenizer=second_tokenizer,
            loss_mask=normal_result["loss_mask"][i],
            new_token_id_threshold=new_token_id_threshold
        )
        viz_strings.append(viz)
        viz_strings_dict[f"{sample_names[i]}, score: {scores[i]}, length: {len(normal_result['input_ids'][i])}"] = viz
        viz_strings.append("\n")
        optimal_tokenization = optimally_tokenize(normal_result["input_ids"][i], second_tokenizer)
        viz_strings.append(f"Optimal_tokenization ({len(optimal_tokenization)} tokens):\n")
        viz2 = visualize_loss_mask(
            input_ids=optimal_tokenization,
            tokenizer=second_tokenizer,
            loss_mask=None,
            new_token_id_threshold=new_token_id_threshold
        )
        viz_strings.append(viz2)
        viz_strings.append("\n")

    if repeat_result is not None:
        viz_strings.append(f"Keys: {repeat_result.keys()}")
        viz_strings.append(f"Input IDs length: {len(repeat_result['input_ids'])}")
        viz_strings.append(f"Loss mask length: {len(repeat_result['loss_mask'])}")
        viz_strings.append("Sample text (detokenized):")
        viz_strings.append("\n=== REPEAT TASK LOSS MASK VISUALIZATION ===")
        for i in range(len(repeat_result["input_ids"])):
            viz_strings.append(f"sample_name: {sample_names[i]} (repeat)")
            viz_strings.append(prompt_list[i])
            viz_strings.append(second_tokenizer.batch_decode(repeat_result["input_ids"][i], skip_special_tokens=False))
            viz = visualize_loss_mask(
                input_ids=repeat_result["input_ids"][i],
                tokenizer=second_tokenizer,
                loss_mask=repeat_result["loss_mask"][i],
            )
            viz_strings.append(viz)
            viz_strings.append("\n")

    print_viz = True
    save_html = False

    if print_viz:
        for viz_string in viz_strings:
            print(viz_string)

    if save_html:
        save_file = "visualizations.html"
        print(f"Saving to {save_file}")
        save_multiple_visualizations_to_html(viz_strings_dict, save_file)
