import torch
import json


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


def initialize_new_embeddings(
    base_embeddings: torch.nn.Parameter,
    num_new_tokens: int,
    init_strategy: str = "default",
    tokenizer = None
) -> torch.nn.Parameter:
    """
    Initialize embeddings for new tokens using different strategies.
    
    Args:
        base_embeddings: Original embedding weights
        num_new_tokens: Number of new tokens to add
        init_strategy: Strategy to use for initialization
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


def extend_model_embeddings(model, num_new_tokens, init_strategy, tokenizer = None):
    """Extend model embeddings to match new tokenizer vocabulary."""
    mean_embedding, mean_lm_head = fix_untrained_tokens(model)  # this is from unsloth, might not be needed
    
    base_vocab_size = model.config.vocab_size
    new_vocab_size = base_vocab_size + num_new_tokens

    # INPUT LAYER
    embedding_matrix = model.get_input_embeddings().weight.data
    new_embeddings_input = initialize_new_embeddings(
        embedding_matrix,
        num_new_tokens,
        init_strategy=init_strategy,
        tokenizer=tokenizer
    )

    # OUTPUT LAYER
    lm_head_matrix   = model.get_output_embeddings().weight.data
    new_embeddings_output = initialize_new_embeddings(
        lm_head_matrix,
        num_new_tokens,
        init_strategy=init_strategy,
        tokenizer=tokenizer
    )

    model.resize_token_embeddings(new_vocab_size)
    embedding_matrix = model.get_input_embeddings ().weight.data
    lm_head_matrix   = model.get_output_embeddings().weight.data

    embedding_matrix[base_vocab_size:] = new_embeddings_input
    lm_head_matrix  [base_vocab_size:] = new_embeddings_output
    
    # Verification
    input_embed_size = model.get_input_embeddings().weight.shape[0]
    output_embed_size = model.get_output_embeddings().weight.shape[0]
    config_vocab_size = model.config.vocab_size

    # print(f"input_embed_size: {input_embed_size}, output_embed_size: {output_embed_size}, config_vocab_size: {config_vocab_size}, new_vocab_size: {new_vocab_size}")
    
    if not (input_embed_size == output_embed_size == config_vocab_size == new_vocab_size):
        raise ValueError(
            f"Embedding extension failed! Sizes don't match:\n"
            f"Expected: {new_vocab_size}\n"
            f"Input embeddings: {input_embed_size}\n"
            f"Output embeddings: {output_embed_size}\n"
            f"Config vocab size: {config_vocab_size}"
        )
    
    # TODO do we need to set a bias term?
    # TODO better to actually create a second embedding layer for the new params that way we can do param grouping here if we need
    
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