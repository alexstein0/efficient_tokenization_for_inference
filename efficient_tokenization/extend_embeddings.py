import torch
import json
from transformers import LlamaForCausalLM

# TODO see class IdeficsDecoupledEmbedding(nn.Embedding):


def get_new_embedding_params(model, num_new_tokens: int):
    """Return a list of parameters corresponding to the new token embeddings.
    Important note: this will not return the grads, only the params.
    """
    
    new_embedding_params = []

    for name, param in model.named_parameters():
        if "embed_tokens.weight" in name or "lm_head.weight" in name:  # Track input/output embeddings
            vocab_size = param.shape[0]  # First dimension is vocab size
            new_embedding_params.append(param[vocab_size - num_new_tokens:])  # Slice only new tokens

    return new_embedding_params  # Returns a list of tensors


def get_new_embeddings_grads(model, num_new_tokens: int):
    """
    Get the embeddings for the newly added tokens.
    
    Args:
        model: The model with extended vocabulary
        num_new_tokens: Number of new tokens that were added
        
    Returns:
        list: List of parameters corresponding to new token embeddings
    """
    # Handle distributed models
    if hasattr(model, 'module'):
        model = model.module
    
    embeddings_output = model.get_output_embeddings()
    embeddings_input = model.get_input_embeddings()
    
    # Create indices for the new tokens
    vocab_size = model.config.vocab_size

    with torch.no_grad():
        grad_slice_output = embeddings_output.weight.grad[vocab_size - num_new_tokens:].clone().detach()
        grad_slice_input = embeddings_input.weight.grad[vocab_size - num_new_tokens:].clone().detach()
        # grad_norm = grad_slice.norm()

    # new_embeddings_grad_output = embeddings_output.weight.grad[vocab_size - num_new_tokens:]
    # new_embeddings_grad_input = embeddings_input.weight.grad[vocab_size - num_new_tokens:]

    # Return as list of parameters
    return [grad_slice_input, grad_slice_output]


def get_old_embedding_params(model, num_new_tokens: int):
    """Return a list of parameters corresponding to the new token embeddings.
    Important note: this will not return the grads, only the params.
    """
    
    new_embedding_params = []

    for name, param in model.named_parameters():
        if "embed_tokens.weight" in name or "lm_head.weight" in name:  # Track input/output embeddings
            vocab_size = param.shape[0]  # First dimension is vocab size
            new_embedding_params.append(param[:vocab_size - num_new_tokens])  # Slice only old tokens

    return new_embedding_params  # Returns a list of tensors

def get_old_embeddings_grads(model, num_new_tokens: int):
    """
    Get the embeddings for the newly added tokens.
    
    Args:
        model: The model with extended vocabulary
        num_new_tokens: Number of new tokens that were added
        
    Returns:
        list: List of parameters corresponding to new token embeddings
    """
    # Handle distributed models
    if hasattr(model, 'module'):
        model = model.module
    
    embeddings_output = model.get_output_embeddings()
    embeddings_input = model.get_input_embeddings()
    
    # Create indices for the new tokens
    vocab_size = model.config.vocab_size

    with torch.no_grad():
        grad_slice_output = embeddings_output.weight.grad[:vocab_size - num_new_tokens].clone().detach()
        grad_slice_input = embeddings_input.weight.grad[:vocab_size - num_new_tokens].clone().detach()

    # new_embeddings_grad_output = embeddings_output.weight.grad[vocab_size - num_new_tokens:]
    # new_embeddings_grad_input = embeddings_input.weight.grad[vocab_size - num_new_tokens:]

    # Return as list of parameters
    return [grad_slice_input, grad_slice_output]

def calculate_new_embeddings_grad_norm(model, num_new_tokens, norm_type=2.0):
    """Calculate the gradient norm of the new embeddings."""
    base_vocab_size = model.config.vocab_size - num_new_tokens
    input_grad = model.get_input_embeddings().weight.grad[base_vocab_size:]
    output_grad = model.get_output_embeddings().weight.grad[base_vocab_size:]
    input_norm = torch.norm(input_grad, p=norm_type)
    output_norm = torch.norm(output_grad, p=norm_type)
    return input_norm, output_norm


def initialize_new_embeddings(
    base_embeddings: torch.Tensor,
    num_new_tokens: int,
    init_strategy: str = "default",
    tokenizer=None
) -> torch.Tensor:
    """
    Initialize embeddings for new tokens using different strategies.
    
    Args:
        base_embeddings: Original embedding weights
        num_new_tokens: Number of new tokens to add
        init_strategy: Strategy to use for initialization
            - "default": HF setting
            - "random": Standard normal initialization
            - "clone": Clone random existing embeddings
            - "mean": Initialize to mean of base embeddings
            - "zeros": Initialize to zeros
    """
    device = base_embeddings.device
    dtype = base_embeddings.dtype  # This will be bfloat16
    embed_dim = base_embeddings.shape[1]
    
    if init_strategy == "default":
        vocab_size = base_embeddings.shape[0]
        new_embeddings = base_embeddings[vocab_size - num_new_tokens:]

    elif init_strategy == "random":
        # Initialize directly with correct dtype
        new_embeddings = torch.empty(num_new_tokens, embed_dim, device=device, dtype=dtype)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=dtype):  # Fixed autocast
            new_embeddings.normal_()
            std = base_embeddings.std().item()
            new_embeddings.mul_(std)
    
    elif init_strategy == "clone":
        # Randomly select tokens to clone
        indices = torch.randint(0, len(base_embeddings), (num_new_tokens,))
        new_embeddings = base_embeddings[indices].clone()
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=dtype):
            noise = torch.randn_like(new_embeddings) * 0.1
            new_embeddings += noise
    
    elif init_strategy == "mean":
        # Use mean of base embeddings
        mean_embedding = base_embeddings.mean(0, keepdim=True)
        # Calculate std across all embeddings (scalar value)
        embeddings_std = torch.std(base_embeddings).item()
        new_embeddings = mean_embedding.repeat(num_new_tokens, 1)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=dtype):
            noise = torch.randn_like(new_embeddings) * embeddings_std
            new_embeddings += noise
    
    elif init_strategy == "zeros":
        new_embeddings = torch.zeros(num_new_tokens, embed_dim, device=device, dtype=dtype)

    elif init_strategy == "merge":
        # TODO: each word is an average of the base embeddings that created that word
        if tokenizer is None:
            raise ValueError("Tokenizer is required for merged initialization strategy")
        
        new_embeddings = torch.zeros(num_new_tokens, embed_dim, device=device, dtype=dtype)

        tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
        new_merge_list = tokenizer_json["model"]["merges"][-num_new_tokens:]
        for i, (first, second) in enumerate(new_merge_list):
            first_id = tokenizer.convert_tokens_to_ids(first)
            second_id = tokenizer.convert_tokens_to_ids(second)
            # print(f"Token {i}: {first} {second} {first_id} {second_id}")
            if first_id < base_embeddings.shape[0]:
                first_embedding = base_embeddings[first_id]
            else:
                first_embedding = new_embeddings[first_id - base_embeddings.shape[0]]
            if second_id < base_embeddings.shape[0]:
                second_embedding = base_embeddings[second_id]
            else:
                second_embedding = new_embeddings[second_id - base_embeddings.shape[0]]
                
            this_embedding = (first_embedding + second_embedding) / 2
            new_embeddings[i] = this_embedding

    else:
        raise ValueError(f"Unknown initialization strategy: {init_strategy}")
    
    return new_embeddings


def extend_model_embeddings(model, num_new_tokens, init_strategy="default", tokenizer=None):
    # 1) Retrieve original embedding weights
    mean_embedding, mean_lm_head = fix_untrained_tokens(model)  # this is from unsloth, might not be needed

    base_vocab_size = model.config.vocab_size
    new_vocab_size = base_vocab_size + num_new_tokens

    embedding_matrix = model.get_input_embeddings().weight.data
    lm_head_matrix = model.get_output_embeddings().weight.data

    # 2) Prepare new embeddings for input & output
    new_emb_input = initialize_new_embeddings(embedding_matrix, num_new_tokens, init_strategy, tokenizer)
    new_emb_output = initialize_new_embeddings(lm_head_matrix, num_new_tokens, init_strategy, tokenizer)

    # 3) Resize the model to reflect new vocab size
    model.resize_token_embeddings(new_vocab_size)

    # After resizing, retrieve the updated embedding pointers
    updated_embed_matrix = model.get_input_embeddings().weight.data
    updated_lm_head_matrix = model.get_output_embeddings().weight.data

    # 4) Fill the new slice with the newly initialized embeddings
    updated_embed_matrix[base_vocab_size:] = new_emb_input
    updated_lm_head_matrix[base_vocab_size:] = new_emb_output

    # 5) Verify shapes
    if (updated_embed_matrix.shape[0] != new_vocab_size
        or updated_lm_head_matrix.shape[0] != new_vocab_size
        or model.config.vocab_size != new_vocab_size):
        raise ValueError(
            f"Embedding extension mismatch:\n"
            f"  Extended to {new_vocab_size} but got:\n"
            f"  input_emb={updated_embed_matrix.shape[0]}, "
            f"  output_emb={updated_lm_head_matrix.shape[0]}, "
            f"  config_vocab={model.config.vocab_size}"
        )
    
    # TODO do we need to set a bias term?
    # TODO is it better to actually create a second embedding layer for the new params that way we can do param grouping here if we need
    
    return model

def my_resize_token_embeddings(model, new_num_tokens, mean_resizing):
    # TODO dont use this either
    old_embeddings = model.get_input_embeddings()
    new_embeddings = model._get_resized_embeddings(
        old_embeddings, new_num_tokens, pad_to_multiple_of = None, mean_resizing = True
    )
    if hasattr(old_embeddings, "_hf_hook"):
        hook = old_embeddings._hf_hook
        from accelerate.hooks import add_hook_to_module
        add_hook_to_module(new_embeddings, hook)

    old_embeddings_requires_grad = old_embeddings.weight.requires_grad
    new_embeddings.requires_grad_(old_embeddings_requires_grad)
    model.set_input_embeddings(new_embeddings)

    # if word embeddings are not tied, make sure that lm head is resized as well
    if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
        old_lm_head = model.get_output_embeddings()
        if isinstance(old_lm_head, torch.nn.Embedding):
            new_lm_head = model._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
        else:
            new_lm_head = model._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
        if hasattr(old_lm_head, "_hf_hook"):
            hook = old_lm_head._hf_hook
            add_hook_to_module(new_lm_head, hook)
        old_lm_head_requires_grad = old_lm_head.weight.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        model.set_output_embeddings(new_lm_head)

    # return self.get_input_embeddings()
    # model_embeds = model._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
    model_embeds = model.get_input_embeddings()

    vocab_size = model_embeds.weight.shape[0]

    # Update base model and current model config.
    model.config.get_text_config().vocab_size = vocab_size
    model.vocab_size = vocab_size

    # Tie weights again if needed
    model.tie_weights()

    return model_embeds


def extend_model_embeddings_with_multi_layer(model, num_new_tokens, init_strategy, tokenizer = None):
    # TODO THIS DOES NOT WORK, PROBABLY EASIER TO DO OTHER WAY
    """Extend model embeddings to match new tokenizer vocabulary."""
    mean_embedding, mean_lm_head = fix_untrained_tokens(model)
    
    base_vocab_size = model.config.vocab_size
    new_vocab_size = base_vocab_size + num_new_tokens
    embed_dim = model.get_input_embeddings().weight.shape[1]
    device = model.get_input_embeddings().weight.device
    dtype = model.get_input_embeddings().weight.dtype

    # INPUT LAYER
    embedding_matrix = model.get_input_embeddings().weight.data
    lm_head_matrix = model.get_output_embeddings().weight.data
    
    new_embeddings_input = initialize_new_embeddings(
        embedding_matrix,
        num_new_tokens,
        init_strategy=init_strategy,
        tokenizer=tokenizer
    )

    # OUTPUT LAYER
    new_embeddings_output = initialize_new_embeddings(
        lm_head_matrix,
        num_new_tokens,
        init_strategy=init_strategy,
        tokenizer=tokenizer
    )

    # Create new embedding layers
    model.base_embeddings = model.get_input_embeddings()
    model.new_embeddings = torch.nn.Embedding(num_new_tokens, embed_dim, device=device, dtype=dtype)
    model.new_embeddings.weight.data = new_embeddings_input

    model.base_lm_head = model.get_output_embeddings()
    model.new_lm_head = torch.nn.Linear(embed_dim, num_new_tokens, bias=False, device=device, dtype=dtype)
    model.new_lm_head.weight.data = new_embeddings_output

    # Create new forward methods as class methods
    class NewEmbeddings(torch.nn.Module):
        def __init__(self, base_embeddings, new_embeddings, base_vocab_size):
            super().__init__()
            self.base_embeddings = base_embeddings
            self.new_embeddings = new_embeddings
            self.base_vocab_size = base_vocab_size
            
        def forward(self, input_ids, *args, **kwargs):
            base_embeds = self.base_embeddings(input_ids.clamp_max(self.base_vocab_size - 1))
            new_ids = (input_ids - self.base_vocab_size).clamp_min(0)
            new_embeds = self.new_embeddings(new_ids)
            embeds = torch.where(
                (input_ids < self.base_vocab_size).unsqueeze(-1),
                base_embeds,
                new_embeds
            )
            return embeds

    class NewLMHead(torch.nn.Module):
        def __init__(self, base_lm_head, new_lm_head):
            super().__init__()
            self.base_lm_head = base_lm_head
            self.new_lm_head = new_lm_head
            
        def forward(self, hidden_states, *args, **kwargs):
            base_logits = self.base_lm_head(hidden_states)
            new_logits = self.new_lm_head(hidden_states)
            return torch.cat([base_logits, new_logits], dim=-1)

    # Replace the embeddings and lm_head with our new modules
    model.embeddings = NewEmbeddings(model.base_embeddings, model.new_embeddings, base_vocab_size)
    model.lm_head = NewLMHead(model.base_lm_head, model.new_lm_head)
    
    # Update the model's forward methods
    model.get_input_embeddings = lambda: model.embeddings
    model.get_output_embeddings = lambda: model.lm_head
    
    # Update model config
    model.config.vocab_size = new_vocab_size
    
    # Verification
    input_embed_size = base_vocab_size + model.new_embeddings.weight.shape[0]
    output_embed_size = base_vocab_size + model.new_lm_head.weight.shape[0]
    config_vocab_size = model.config.vocab_size

    if not (input_embed_size == output_embed_size == config_vocab_size == new_vocab_size):
        raise ValueError(
            f"Embedding extension failed! Sizes don't match:\n"
            f"Expected: {new_vocab_size}\n"
            f"Input embeddings: {input_embed_size}\n"
            f"Output embeddings: {output_embed_size}\n"
            f"Config vocab size: {config_vocab_size}"
        )
    
    # TODO do we need to set a bias term?
    
    return model


# from unsloth:
def fix_untrained_tokens(model, eps = 1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings().weight.data
    lm_head_matrix   = model.get_output_embeddings().weight.data

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    if n_untrained != 0:
        print(
            f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
            "We shall set them to the mean of the other trained tokens."
        )

    # First set untrained to all 0s - sometimes it's not! 1e-23 for bfloat16
    embedding_matrix[where_untrained] = 0
    lm_head_matrix  [where_untrained] = 0

    # Find sum
    sum_embedding  = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head    = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained).to(embedding_matrix.dtype)
    mean_lm_head   = (sum_lm_head   / n_trained).to(lm_head_matrix  .dtype)

    # Set them to the mean
    embedding_matrix[where_untrained] = mean_embedding
    lm_head_matrix  [where_untrained] = mean_lm_head

    return mean_embedding, mean_lm_head

def freeze_old_embeddings(model, num_new_tokens):
    """Freeze the old embeddings, allowing only the new ones to be trained."""
    base_vocab_size = model.config.vocab_size - num_new_tokens
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = False
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = False

    # Unfreeze the new embeddings
    model.get_input_embeddings().weight.requires_grad[base_vocab_size:] = True
    model.get_output_embeddings().weight.requires_grad[base_vocab_size:] = True
    return model

def freeze_model_except_embeddings(model, freeze_output_embeddings=True):
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model = unfreeze_embeddings(model, freeze_output_embeddings=freeze_output_embeddings)
    
    return model

def unfreeze_embeddings(model, freeze_output_embeddings=True):
    # Unfreeze input embeddings
    if hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings().weight.requires_grad = True
    
    # Optionally unfreeze output embeddings if they're not tied
    if freeze_output_embeddings and hasattr(model, 'get_output_embeddings'):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight.requires_grad = True
    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    
    return model

def unfreeze_first_last_layer(model, logger):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        if len(model.model.layers) > 0:
            for param in model.model.layers[0].parameters():
                param.requires_grad = True
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True
        else:
            logger.warning("No layers found in model.model.layers!")
    else:
        logger.warning("Model does not have the expected Llama 'model.layers' structure.")
    return model


        
# Example usage:
# model = ...  # Your model
# num_new_tokens = 100
# model = extend_model_embeddings(model, num_new_tokens, init_strategy="mean", tokenizer=tokenizer)
# freeze_old_embeddings(model, num_new_tokens)
# # After a backward pass
# input_norm, output_norm = calculate_new_embeddings_grad_norm(model, num_new_tokens)