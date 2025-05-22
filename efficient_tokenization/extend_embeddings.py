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
    import_path=None,
    logger=None
) -> tuple[torch.Tensor, torch.Tensor|None]:
    """
    Initialize embeddings for new tokens using different strategies.
    
    Args:
        base_embeddings: Original embedding weights (already resized)
        num_new_tokens: Number of new tokens to add
        init_strategy: Strategy to use for initialization
            - "default": HF setting
            - "random": Standard normal initialization
            - "clone": Clone random existing embeddings
            - "mean": Initialize to mean of base embeddings
            - "zeros": Initialize to zeros
    Returns:
        new_embeddings: New embeddings for the new tokens
        output_embeddings: New embeddings for the output tokens
    """
    # Important to note that this is already resized, the reason being is that sometimes the model has extra embeddings that arent in the tokenizer, so we already need to slice only the old embeddings range rather than dummy values
    device = base_embeddings.device
    dtype = base_embeddings.dtype
    embed_dim = base_embeddings.shape[1]
    new_vocab_size = len(tokenizer)
    new_token_start = new_vocab_size - num_new_tokens  # this is the new token start in the VOCAB note that the merges might not map to the vocab 1 to 1
    new_token_range = slice(new_token_start, new_vocab_size)
    old_token_range = slice(0, new_token_start)

    old_vocab_embeddings = base_embeddings[old_token_range]
    
    assert len(tokenizer) <= base_embeddings.shape[0], f"Tokenizer has {len(tokenizer)} tokens, but base embeddings have {base_embeddings.shape[0]} tokens"
    if logger is not None:
        logger.info(f"Initializing {num_new_tokens} new embeddings for token range {new_token_range} using strategy {init_strategy} {import_path if import_path is not None else ''}", main_process_only=True)

    output_embeddings = None
    if init_strategy == "default":
        new_embeddings = base_embeddings[new_token_range]

    elif init_strategy == "random":
        # Initialize directly with correct dtype
        new_embeddings = torch.empty(num_new_tokens, embed_dim, device=device, dtype=dtype)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=dtype):  # Fixed autocast
            new_embeddings.normal_()
            std = old_vocab_embeddings.std().item()
            new_embeddings.mul_(std)
    
    elif init_strategy == "clone":
        # Randomly select tokens to clone
        indices = torch.randint(0, new_token_start, (num_new_tokens,))
        new_embeddings = base_embeddings[indices].clone()
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=dtype):
            noise = torch.randn_like(new_embeddings) * 0.1
            new_embeddings += noise
    
    elif init_strategy == "mean":
        # Use mean of base embeddings
        mean_embedding = old_vocab_embeddings.mean(0, keepdim=True)
        # Calculate std across all embeddings (scalar value)
        embeddings_std = torch.std(old_vocab_embeddings).item()
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
        new_merge_list = tokenizer_json["model"]["merges"][-num_new_tokens:]  # this needs to get the last x merges bc that doesnt correspond to the new tokens for sure
        for i, (first, second) in enumerate(new_merge_list):
            first_id = tokenizer.convert_tokens_to_ids(first)
            second_id = tokenizer.convert_tokens_to_ids(second)
            # print(f"Token {i}: {first} {second} {first_id} {second_id}")
            if first_id < new_token_start:
                first_embedding = base_embeddings[first_id]
            else:
                first_embedding = new_embeddings[first_id - new_token_start]
            if second_id < new_token_start:
                second_embedding = base_embeddings[second_id]
            else:
                second_embedding = new_embeddings[second_id - new_token_start]
                
            this_embedding = (first_embedding + second_embedding) / 2
            new_embeddings[i] = this_embedding

    elif init_strategy == "import":
        if import_path is None:
            raise ValueError("import_path is required for 'import' initialization strategy")

        embeddings_path = os.path.join(import_path, "embeddings_only.pt")
        if os.path.exists(embeddings_path):
            try:
                # TODO MAKE IT SO WE ONLY ADD THE NEW TOKENS
                embeddings = torch.load(embeddings_path, map_location="cpu")
                loaded_input_emb = embeddings["input_embeddings"][new_token_range].clone().to(base_embeddings.device, dtype=base_embeddings.dtype)
                loaded_output_emb = embeddings["output_embeddings"][new_token_range].clone().to(base_embeddings.device, dtype=base_embeddings.dtype)
                return loaded_input_emb, loaded_output_emb
            except Exception as e:
                print(f"Error loading embeddings directly from file {embeddings_path}, trying from saved model")

        from safetensors import safe_open
        if os.path.isdir(import_path):
            safetensor_files = sorted(
                [os.path.join(import_path, f) for f in os.listdir(import_path) if f.endswith(".safetensors")]
            )
        else:
            safetensor_files = [import_path]

        if len(safetensor_files) == 0:
            print(f"{import_path}: No safetensors files found")
            return None, None

        param_dict = {}
        for file in safetensor_files:
            with safe_open(file, framework="pt") as f:
                for key in f.keys():
                    param_dict[key] = f.get_tensor(key)

        try:
            loaded_embeddings = param_dict["model.embed_tokens.weight"].cpu()
        except:
            print(f"{import_path}: model.embed_tokens.weight not found in safetensors files. Options: {param_dict.keys()}")
            return None, None

        try:
            output_embeddings = param_dict["lm_head.weight"].cpu()
        except:
            # print(f"{model_path}: lm_head.weight not found in safetensors files. Options: {param_dict.keys()}")
            output_embeddings = loaded_embeddings.clone()

        if loaded_embeddings.shape[1] != embed_dim:
            raise ValueError(f"Imported embeddings have dimension {loaded_embeddings.shape[1]}, expected {embed_dim}")

        if loaded_embeddings.shape[0] < base_embeddings.shape[0]:
            raise ValueError(f"Imported embeddings have {loaded_embeddings.shape[0]} tokens, but {base_embeddings.shape[0]} new tokens requested")
        
        new_embeddings = loaded_embeddings[new_token_range].clone().to(base_embeddings.device, dtype=base_embeddings.dtype)
        new_output_embeddings = output_embeddings[new_token_range].clone().to(base_embeddings.device, dtype=base_embeddings.dtype)

        return new_embeddings, new_output_embeddings

    else:
        raise ValueError(f"Unknown initialization strategy: {init_strategy}")
    
    return new_embeddings, output_embeddings

def extend_model_embeddings(model, num_new_tokens, init_strategy="default", tokenizer=None, import_path=None, logger=None):
    mean_embedding, mean_lm_head = fix_untrained_tokens(model)  # this is from unsloth, might not be needed
    # 1) resize embeddings if needed
    new_vocab_size = len(tokenizer)
    old_embedding_size = model.config.vocab_size
    new_tokens_range = slice(old_embedding_size, new_vocab_size)

    # determine how big the embedding layer needs to be
    # sometimes the model has extra embeddings that arent in the tokenizer, so we dont need to add all of them to embedding layer
    if new_vocab_size > old_embedding_size:
        model.resize_token_embeddings(new_vocab_size)
    else:
        # models like phi have extra random embeddings that arent in the tokenizer
        if logger is not None:
            logger.info(f"Model has already extended embeddings with new tokens so no need to resize embeddings, but still need to initialize {num_new_tokens} new embeddings", main_process_only=True)
    
    # 2) Retrieve original embedding weights
    embedding_matrix = model.get_input_embeddings().weight.data
    lm_head_matrix = model.get_output_embeddings().weight.data

    # 3) Prepare new embeddings for input & output for the new tokens
    new_emb_input, new_emb_output  = initialize_new_embeddings(embedding_matrix, num_new_tokens, init_strategy, tokenizer, import_path, logger)
    if new_emb_output is None:
        new_emb_output, _ = initialize_new_embeddings(lm_head_matrix, num_new_tokens, init_strategy, tokenizer, import_path, logger)

    # 4) Fill the new slice with the newly initialized embeddings
    embedding_matrix[new_tokens_range] = new_emb_input
    lm_head_matrix[new_tokens_range] = new_emb_output

    # 5) Verify shapes
    assert embedding_matrix.shape[0] == max(new_vocab_size, old_embedding_size), f"Embedding matrix has {embedding_matrix.shape[0]} tokens, expected {max(new_vocab_size, old_embedding_size)}"
    assert lm_head_matrix.shape[0] == max(new_vocab_size, old_embedding_size), f"LM head matrix has {lm_head_matrix.shape[0]} tokens, expected {max(new_vocab_size, old_embedding_size)}"
    assert model.config.vocab_size == max(new_vocab_size, old_embedding_size), f"Model config has {model.config.vocab_size} tokens, expected {max(new_vocab_size, old_embedding_size)}"
    assert new_emb_input.shape[0] == num_new_tokens, f"New input embeddings have {new_emb_input.shape[0]} tokens, expected {num_new_tokens}"
    assert new_emb_output.shape[0] == num_new_tokens, f"New output embeddings have {new_emb_output.shape[0]} tokens, expected {num_new_tokens}"

    return model


# def extend_model_embeddings_with_multi_layer(model, num_new_tokens, init_strategy, tokenizer = None, import_path = None):
#     # TODO THIS DOES NOT WORK, PROBABLY EASIER TO DO OTHER WAY
#     """Extend model embeddings to match new tokenizer vocabulary."""
#     mean_embedding, mean_lm_head = fix_untrained_tokens(model)
    
#     base_vocab_size = model.config.vocab_size
#     new_vocab_size = base_vocab_size + num_new_tokens
#     embed_dim = model.get_input_embeddings().weight.shape[1]
#     device = model.get_input_embeddings().weight.device
#     dtype = model.get_input_embeddings().weight.dtype

#     # INPUT LAYER
#     embedding_matrix = model.get_input_embeddings().weight.data
#     lm_head_matrix = model.get_output_embeddings().weight.data
    
#     new_embeddings_input = initialize_new_embeddings(
#         embedding_matrix,
#         num_new_tokens,
#         init_strategy=init_strategy,
#         tokenizer=tokenizer,
#         import_path=import_path
#     )

#     # OUTPUT LAYER
#     new_embeddings_output = initialize_new_embeddings(
#         lm_head_matrix,
#         num_new_tokens,
#         init_strategy=init_strategy,
#         tokenizer=tokenizer,
#         import_path=import_path
#     )

#     # Create new embedding layers
#     model.base_embeddings = model.get_input_embeddings()
#     model.new_embeddings = torch.nn.Embedding(num_new_tokens, embed_dim, device=device, dtype=dtype)
#     model.new_embeddings.weight.data = new_embeddings_input

#     model.base_lm_head = model.get_output_embeddings()
#     model.new_lm_head = torch.nn.Linear(embed_dim, num_new_tokens, bias=False, device=device, dtype=dtype)
#     model.new_lm_head.weight.data = new_embeddings_output

#     # Create new forward methods as class methods
#     class NewEmbeddings(torch.nn.Module):
#         def __init__(self, base_embeddings, new_embeddings, base_vocab_size):
#             super().__init__()
#             self.base_embeddings = base_embeddings
#             self.new_embeddings = new_embeddings
#             self.base_vocab_size = base_vocab_size
            
#         def forward(self, input_ids, *args, **kwargs):
#             base_embeds = self.base_embeddings(input_ids.clamp_max(self.base_vocab_size - 1))
#             new_ids = (input_ids - self.base_vocab_size).clamp_min(0)
#             new_embeds = self.new_embeddings(new_ids)
#             embeds = torch.where(
#                 (input_ids < self.base_vocab_size).unsqueeze(-1),
#                 base_embeds,
#                 new_embeds
#             )
#             return embeds

#     class NewLMHead(torch.nn.Module):
#         def __init__(self, base_lm_head, new_lm_head):
#             super().__init__()
#             self.base_lm_head = base_lm_head
#             self.new_lm_head = new_lm_head
            
#         def forward(self, hidden_states, *args, **kwargs):
#             base_logits = self.base_lm_head(hidden_states)
#             new_logits = self.new_lm_head(hidden_states)
#             return torch.cat([base_logits, new_logits], dim=-1)

#     # Replace the embeddings and lm_head with our new modules
#     model.embeddings = NewEmbeddings(model.base_embeddings, model.new_embeddings, base_vocab_size)
#     model.lm_head = NewLMHead(model.base_lm_head, model.new_lm_head)
    
#     # Update the model's forward methods
#     model.get_input_embeddings = lambda: model.embeddings
#     model.get_output_embeddings = lambda: model.lm_head
    
#     # Update model config
#     model.config.vocab_size = new_vocab_size
    
#     # Verification
#     input_embed_size = base_vocab_size + model.new_embeddings.weight.shape[0]
#     output_embed_size = base_vocab_size + model.new_lm_head.weight.shape[0]
#     config_vocab_size = model.config.vocab_size

#     if not (input_embed_size == output_embed_size == config_vocab_size == new_vocab_size):
#         raise ValueError(
#             f"Embedding extension failed! Sizes don't match:\n"
#             f"Expected: {new_vocab_size}\n"
#             f"Input embeddings: {input_embed_size}\n"
#             f"Output embeddings: {output_embed_size}\n"
#             f"Config vocab size: {config_vocab_size}"
#         )
    
#     # TODO do we need to set a bias term?
    
#     return model


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