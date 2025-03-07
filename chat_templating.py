from typing import Union, List, Dict, Optional, Any

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TensorType
from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices
import re
import copy



def get_llama_base_chat_template():
    fake_system_prompt = """You are a helpful AI assistant well versed in repeating the message back to them in an equivalent but more efficient way. The original message will be indicated by 'text to repeat:' and end with 'end text.'  The repeat section will be indicated by 'repeat:'"""
    template_list = [
        {
            "name": "base_template",
            "tokenizer": "base",
            "template": f"""<|begin_of_text|>{fake_system_prompt}\n"""
        },
        {
            "name": "original_message_template",
            "tokenizer": "base",
            "template": """text to repeat: {{ messages[0].content }} end text. \nrepeat: """
        },
        {
            "name": "new_message_template",
            "tokenizer": "second",
            "template": """{% generation %}{{ messages[0].content }}{% endgeneration %}"""
        },
        
    ]
    return template_list

# example templates from llama
def get_llama_instruct_chat_template():
    #"chat_template" = "{% set loop_messages = messages %}
    # {% for message in loop_messages %}
    # {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}
    # {% if loop.index0 == 0 %}
    # {% set content = bos_token + content %}
    # {% endif %}
    # {{ content }}
    # {% endfor %}
    # {% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    template_list = [
        {
            "name": "base_template",
            "tokenizer": "base",
            "template": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>}

    You are a helpful AI assistant well versed in repeating the user's question back to them in a different but equivalent way.<|eot_id|>"""
        },
        {
            "name": "original_message_template",
            "tokenizer": "base",
            "template": """<|start_header_id|>user<|end_header_id|>

    {{ message.content }}<|eot_id|><|start_header_id|>assistent<|end_header_id|>"""
        },
        {
            "name": "new_message_template",
            "tokenizer": "second",
            "template": """{% generation %}{{ message.content }}{% endgeneration %}"""
        },
        {
            "name": "end_message_template",
            "tokenizer": "base",
            "template": """<|eot_id|>"""
        } 
    ]
    return template_list

def apply_chat_template_to_repeat(
        base_tokenizer,
        second_tokenizer,
        # conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        conversation: str | List[str],  # just the string to be used as the message (repeated)
        # tools: Optional[List[Dict]] = None,
        # documents: Optional[List[Dict[str, str]]] = None,
        chat_template:  str|List[Dict[str, str]], # not optional anymore
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        # padding: bool = False,
        # truncation: bool = False,
        # max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding, List[Dict[str, List]]]:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
        determine the format and control tokens to use when converting.

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (`List[Dict]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.
            documents (`List[Dict[str, str]]`, *optional*):
                A list of dicts representing documents that will be accessible to the model if it is performing RAG
                (retrieval-augmented generation). If the template does not support RAG, this argument will have no
                effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
                see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
                for examples of passing documents with chat templates.
            chat_template (`str`, *optional*):
                A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
                argument, as the model's template will be used by default.
            add_generation_prompt (bool, *optional*):
                If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final
                message in the chat is open-ended, without any EOS tokens. The model will continue this message
                rather than starting a new one. This allows you to "prefill" part of
                the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            return_assistant_tokens_mask (`bool`, defaults to `False`):
                Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
                the mask will contain 1. For user and system tokens, the mask will contain 0.
                This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `Union[List[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
            set, will return a dict of tokenizer outputs instead.
        """

        if return_dict and not tokenize:
            raise ValueError(
                "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
                "of tokenizer outputs to return."
            )

        if return_assistant_tokens_mask and not return_dict:
            raise ValueError("`return_assistant_tokens_mask=True` is incompatible with `return_dict=False`")

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        # chat_template = self.get_chat_template(chat_template, tools)
        # Compilation function uses a cache to avoid recompiling the same template
        if isinstance(chat_template, list):
            chat_template = "".join([t["template"] for t in chat_template])

        if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
            # logger.warning_once(
            print(
                "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
            )

        compiled_template = _compile_jinja_template(chat_template)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        tool_schemas = None
        documents = None

        rendered = []
        all_generation_indices = []
        template_kwargs = {**base_tokenizer.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present

        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            if return_assistant_tokens_mask:
                rendered_chat, generation_indices = _render_with_assistant_indices(
                    compiled_template=compiled_template,
                    messages=chat,
                    tools=tool_schemas,
                    documents=documents,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
                all_generation_indices.append(generation_indices)
            else:
                rendered_chat = compiled_template.render(
                    messages=chat,
                    tools=tool_schemas,
                    documents=documents,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
            if continue_final_message:
                final_message = chat[-1]["content"]
                if isinstance(final_message, (list, tuple)):
                    final_message = final_message[-1]["text"]
                try:
                    rendered_chat = rendered_chat[: rendered_chat.rindex(final_message) + len(final_message)]
                except:  # noqa: E722
                    # Some chat templates like Llama-3.1 trim messages before rendering, so we must do the same here.
                    final_message = final_message.strip()
                    rendered_chat = rendered_chat[: rendered_chat.rindex(final_message) + len(final_message)]
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        texts = []
        input_ids = []
        attention_mask = []
        labels = []
        assistant_masks = []

        if tokenize:
            for i in range(len(rendered)):
                # only one generation index per conversation
                base_text = rendered[i][:all_generation_indices[i][0][0]]
                generated_text = rendered[i][all_generation_indices[i][0][0]:all_generation_indices[i][0][1]]
                rest_of_text = rendered[i][all_generation_indices[i][0][1]:]

                out_base = base_tokenizer(
                    base_text,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs,
                )
                out_generated = second_tokenizer(
                    generated_text,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs,
                )
                out_rest = base_tokenizer(
                    rest_of_text,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs,
                )
                # TODO return batch encoding instead of dict
                
                texts.append(rendered[i])
                input_ids.append(out_base["input_ids"] + out_generated["input_ids"] + out_rest["input_ids"])
                attention_mask.append(out_base["attention_mask"] + out_generated["attention_mask"] + out_rest["attention_mask"])
                assistant_masks.append([0] * len(out_base["input_ids"]) + [1] * len(out_generated["input_ids"]) + [0] * len(out_rest["input_ids"]))
                labels.append(input_ids[-1].copy())

            return {"text": texts, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "loss_mask": assistant_masks}
        else:
            return {"text": rendered}


if __name__ == "__main__":
    base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    second_tokenizer = AutoTokenizer.from_pretrained("tokenizers/Llama-3.2-tokenizer-genqa-math-empty-start-1000")
    
    # Format the conversation as a list of messages
    conversation = [
        [{
            "role": "user",
            "content": "Hello, how are you?"
        }],
        [{
            "role": "user",
            "content": "CONVO 2"
        }]
    ]
    
    chat_template = get_llama_base_chat_template()
    
    result = apply_chat_template_to_repeat(
        base_tokenizer=base_tokenizer,
        second_tokenizer=second_tokenizer,
        conversation=conversation,
        chat_template=chat_template,
        return_assistant_tokens_mask=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    
    print("Result:", result)