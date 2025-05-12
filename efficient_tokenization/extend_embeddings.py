import torch
import json
import os
# TODO see class IdeficsDecoupledEmbedding(nn.Embedding):



def get_new_embeddings(model, num_new_tokens: int):
    # note, this gets for logging purposes only
    """Return new token embedding weights and their gradients as (params, grads)."""
    if hasattr(model, 'module'):
        model = model.module

    embeddings_output = model.get_output_embeddings()
    embeddings_input = model.get_input_embeddings()

    vocab_size = model.config.vocab_size
    input_slice = slice(vocab_size - num_new_tokens, vocab_size)
    output_slice = slice(vocab_size - num_new_tokens, vocab_size)

    input_weights = embeddings_input.weight[input_slice].clone()
    output_weights = embeddings_output.weight[output_slice].clone()

    with torch.no_grad():
        if embeddings_input.weight.grad is not None:
            input_grads = embeddings_input.weight.grad[input_slice].clone().detach()
        else:
            input_grads = None
        if embeddings_output.weight.grad is not None:
            output_grads = embeddings_output.weight.grad[output_slice].clone().detach()
        else:
            output_grads = None

    return [input_weights, output_weights], [input_grads, output_grads]

def get_old_embeddings(model, num_new_tokens: int):
    # note, this gets for logging purposes only
    """Return old token embedding weights and their gradients as (params, grads)."""
    if hasattr(model, 'module'):
        model = model.module

    embeddings_output = model.get_output_embeddings()
    embeddings_input = model.get_input_embeddings()

    vocab_size = model.config.vocab_size
    input_slice = slice(0, vocab_size - num_new_tokens)
    output_slice = slice(0, vocab_size - num_new_tokens)

    input_weights = embeddings_input.weight[input_slice].cpu().clone()
    output_weights = embeddings_output.weight[output_slice].cpu().clone()

    with torch.no_grad():
        input_grads = embeddings_input.weight.grad[input_slice].cpu().clone().detach()
        output_grads = embeddings_output.weight.grad[output_slice].cpu().clone().detach()

    return [input_weights, output_weights], [input_grads, output_grads]


def initialize_new_embeddings(
    base_embeddings: torch.Tensor,
    num_new_tokens: int,
    init_strategy: str = "default",
    tokenizer=None,
    import_path=None
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

    elif init_strategy == "import":
        if import_path is None:
            raise ValueError("import_path is required for 'import' initialization strategy")

        from safetensors import safe_open

        # Load all safetensors files from directory or use single file
        if os.path.isdir(import_path):
            safetensor_files = sorted(
                [os.path.join(import_path, f) for f in os.listdir(import_path) if f.endswith(".safetensors")]
            )
        else:
            safetensor_files = [import_path]

        param_dict = {}
        for file in safetensor_files:
            with safe_open(file, framework="pt") as f:
                for key in f.keys():
                    param_dict[key] = f.get_tensor(key)

        # Try to find embedding weights by common names
        for key in ["model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight", "weight"]:
            if key in param_dict:
                old_embeddings = param_dict[key]
                break
        else:
            raise ValueError(f"No known embedding key found in safetensors files. Available keys: {list(param_dict.keys())}")

        if old_embeddings.shape[1] != base_embeddings.shape[1]:
            raise ValueError(f"Imported embeddings have dimension {old_embeddings.shape[1]}, expected {base_embeddings.shape[1]}")

        if old_embeddings.shape[0] < num_new_tokens:
            raise ValueError(f"Imported embeddings have {old_embeddings.shape[0]} tokens, but {num_new_tokens} new tokens requested")

        # Use last num_new_tokens embeddings
        new_embeddings = old_embeddings[-num_new_tokens:].clone().to(base_embeddings.device, dtype=base_embeddings.dtype)
    else:
        raise ValueError(f"Unknown initialization strategy: {init_strategy}")
    
    return new_embeddings


def extend_model_embeddings(model, num_new_tokens, init_strategy="default", tokenizer=None, import_path=None):
    # 1) Retrieve original embedding weights
    mean_embedding, mean_lm_head = fix_untrained_tokens(model)  # this is from unsloth, might not be needed

    base_vocab_size = model.config.vocab_size
    new_vocab_size = base_vocab_size + num_new_tokens

    embedding_matrix = model.get_input_embeddings().weight.data
    lm_head_matrix = model.get_output_embeddings().weight.data

    # 2) Prepare new embeddings for input & output
    new_emb_input = initialize_new_embeddings(embedding_matrix, num_new_tokens, init_strategy, tokenizer, import_path)
    new_emb_output = initialize_new_embeddings(lm_head_matrix, num_new_tokens, init_strategy, tokenizer, import_path)

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


def extend_model_embeddings_with_multi_layer(model, num_new_tokens, init_strategy, tokenizer = None, import_path = None):
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
        tokenizer=tokenizer,
        import_path=import_path
    )

    # OUTPUT LAYER
    new_embeddings_output = initialize_new_embeddings(
        lm_head_matrix,
        num_new_tokens,
        init_strategy=init_strategy,
        tokenizer=tokenizer,
        import_path=import_path
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
    """Freeze the old embeddings by zeroing out their gradients manually during training."""
    model = freeze_model_except_embeddings(model)
    base_vocab_size = model.config.vocab_size - num_new_tokens

    # Ensure requires_grad = True for the whole embedding matrix
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = True

    # Register backward hooks to zero out gradients of the old embeddings
    def zero_old_grads_input(grad):
        grad[:base_vocab_size] = 0
        return grad

    def zero_old_grads_output(grad):
        grad[:base_vocab_size] = 0
        return grad

    model.get_input_embeddings().weight.register_hook(zero_old_grads_input)
    model.get_output_embeddings().weight.register_hook(zero_old_grads_output)

    return model

def freeze_model_except_embeddings(model, unfreeze_output_embeddings=True):
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model = unfreeze_embeddings(model, unfreeze_output_embeddings=unfreeze_output_embeddings)
    
    return model

def unfreeze_embeddings(model, unfreeze_output_embeddings=True):
    # Unfreeze input embeddings
    if hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings().weight.requires_grad = True
    
    # Optionally unfreeze output embeddings if they're not tied
    if unfreeze_output_embeddings and hasattr(model, 'get_output_embeddings'):
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