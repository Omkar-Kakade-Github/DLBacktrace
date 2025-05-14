import numpy as np
import torch

def calculate_wt_fc_cuda(wts, inp, w, b, act, device=None):
    """
    Calculate the relevance propagation of a linear layer using CUDA acceleration.
    
    Parameters:
    -----------
    wts : torch.Tensor
        Weights for relevance calculation.
    inp : torch.Tensor
        Input values.
    w : torch.Tensor
        Weight tensor of the layer.
    b : torch.Tensor
        Bias tensor of the layer.
    act : dict
        Activation function details.
    device : torch.device
        Device to use for computation (None will use current device or default to CUDA if available).
        
    Returns:
    --------
    torch.Tensor
        Weighted matrix for relevance propagation.
    """
    # Determine device to use
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure inputs are torch tensors on the correct device
    if not isinstance(wts, torch.Tensor):
        wts = torch.tensor(wts, device=device)
    else:
        wts = wts.to(device)
        
    if not isinstance(inp, torch.Tensor):
        inp = torch.tensor(inp, device=device)
    else:
        inp = inp.to(device)
    
    # Move model parameters to the correct device
    w = w.to(device)
    b = b.to(device)
    
    # Transpose weights to match the original numpy implementation
    w_t = w.T
    
    # Calculate multiplication matrix efficiently (equivalent to einsum)
    mul_mat = torch.matmul(torch.diag(inp), w_t).T
    
    # Initialize weighted matrix
    wt_mat = torch.zeros_like(mul_mat)
    
    # Pre-compute positive and negative indices for each row
    pos_indices = mul_mat > 0
    neg_indices = mul_mat < 0
    
    # Pre-compute positive and negative sums for each row
    p_sums = torch.sum(mul_mat * pos_indices, dim=1)
    n_sums = -torch.sum(mul_mat * neg_indices, dim=1)
    
    # Split bias into positive and negative components
    p_bias = torch.clamp(b, min=0)
    n_bias = -torch.clamp(b, max=0)
    
    # Calculate total sum
    t_sums = p_sums + p_bias - n_sums - n_bias
    
    # Initialize aggregation weights
    p_agg_wts = torch.zeros_like(p_sums)
    n_agg_wts = torch.zeros_like(n_sums)
    
    # Process activation constraints
    if act["type"] == "mono":
        # Apply lower bound constraint
        if act["range"]["l"] is not None:
            mask = t_sums < act["range"]["l"]
            p_sums = torch.where(mask, torch.zeros_like(p_sums), p_sums)
            
        # Apply upper bound constraint
        if act["range"]["u"] is not None:
            mask = t_sums > act["range"]["u"]
            n_sums = torch.where(mask, torch.zeros_like(n_sums), n_sums)
            
    elif act["type"] == "non_mono":
        # Apply activation function.
        # It's expected that act["func"] is a PyTorch-compatible function
        # that operates directly on tensors.
        if callable(act["func"]):
            try:
                # Attempt to use the function directly on tensors
                t_acts = act["func"](t_sums)
                p_acts = act["func"](p_sums + p_bias)
                n_acts = act["func"](-1 * (n_sums + n_bias))
            except Exception as e:
                # Fallback to element-wise computation if direct tensor operation fails
                # This might happen if act["func"] is a Python scalar function.
                # A warning is printed to highlight this performance-critical fallback.
                print(f"Warning: act['func'] '{act['func'].__name__ if hasattr(act['func'], '__name__') else act['func']}' caused an error during direct tensor application: {e}. Falling back to slower element-wise processing. For optimal performance, ensure act['func'] is PyTorch tensor-compatible.")
                t_acts = torch.tensor([act["func"](val.item()) for val in t_sums], device=device)
                p_acts = torch.tensor([act["func"](val.item()) for val in (p_sums + p_bias)], device=device)
                n_acts = torch.tensor([act["func"](val.item()) for val in (-1 * (n_sums + n_bias))], device=device)
        else:
            # This case should ideally not be hit if 'act' dict is formed correctly.
            print(f"Error: act['func'] is not callable for non_mono activation. Value: {act.get('func')}")
            # Or raise ValueError("act['func'] is not callable for non_mono activation in calculate_wt_fc_cuda.")
            # As a last resort, create zero tensors to avoid crashing, though this indicates a setup problem.
            t_acts = torch.zeros_like(t_sums)
            p_acts = torch.zeros_like(p_sums)
            n_acts = torch.zeros_like(n_sums)
        
        # Apply range constraints
        if act["range"]["l"] is not None:
            mask = t_sums < act["range"]["l"]
            p_sums = torch.where(mask, torch.zeros_like(p_sums), p_sums)
            
        if act["range"]["u"] is not None:
            mask = t_sums > act["range"]["u"]
            n_sums = torch.where(mask, torch.zeros_like(n_sums), n_sums)
        
        # Check activation equality conditions
        mask_p_n_positive = (p_sums > 0) & (n_sums > 0)
        mask_t_eq_p = t_acts == p_acts
        mask_t_eq_n = t_acts == n_acts
        
        # Zero out based on activation equality
        p_sums = torch.where(mask_p_n_positive & mask_t_eq_n, torch.zeros_like(p_sums), p_sums)
        n_sums = torch.where(mask_p_n_positive & mask_t_eq_p, torch.zeros_like(n_sums), n_sums)
    
    # Calculate positive aggregation weights
    mask_p_positive = p_sums > 0
    denominators = p_sums + n_sums + p_bias + n_bias
    
    # Avoid division by zero in denominators
    safe_denominators = torch.where(denominators == 0, torch.ones_like(denominators), denominators)
    
    p_term1 = torch.zeros_like(p_sums)
    p_term1[mask_p_positive] = (p_sums[mask_p_positive] + p_bias[mask_p_positive]) / safe_denominators[mask_p_positive]
    
    # Calculate second term carefully to avoid division by zero
    p_sums_plus_bias = p_sums + p_bias
    safe_p_sums_plus_bias = torch.where(p_sums_plus_bias == 0, torch.ones_like(p_sums_plus_bias), p_sums_plus_bias)
    
    p_term2 = p_sums / safe_p_sums_plus_bias
    p_agg_wts = p_term1 * p_term2
    
    # Calculate negative aggregation weights
    mask_n_positive = n_sums > 0
    n_term1 = torch.zeros_like(n_sums)
    n_term1[mask_n_positive] = (n_sums[mask_n_positive] + n_bias[mask_n_positive]) / safe_denominators[mask_n_positive]
    
    # Calculate second term carefully
    n_sums_plus_bias = n_sums + n_bias
    safe_n_sums_plus_bias = torch.where(n_sums_plus_bias == 0, torch.ones_like(n_sums_plus_bias), n_sums_plus_bias)
    
    n_term2 = n_sums / safe_n_sums_plus_bias
    n_agg_wts = n_term1 * n_term2
    
    # Avoid division by zero
    p_sums_safe = torch.where(p_sums == 0, torch.ones_like(p_sums), p_sums)
    n_sums_safe = torch.where(n_sums == 0, torch.ones_like(n_sums), n_sums)
    
    # Vectorized calculation of wt_mat
    # Initialize wt_mat with zeros
    wt_mat = torch.zeros_like(mul_mat)

    # Calculate scaling factors for all rows at once
    # Ensure that division by zero is handled by p_sums_safe and n_sums_safe
    # (they have 1s where original sums were 0)
    pos_scales = (wts * p_agg_wts) / p_sums_safe 
    neg_scales = (wts * n_agg_wts * -1.0) / n_sums_safe

    # Unsqueeze scales to enable broadcasting with mul_mat: (out_features, 1)
    # This applies each row's scale to all elements in that row of mul_mat.
    # These are potential contributions before masking.
    potential_pos_contrib = mul_mat * pos_scales.unsqueeze(1)
    potential_neg_contrib = mul_mat * neg_scales.unsqueeze(1)

    # Assign contributions to wt_mat only where the original indices were positive/negative
    # pos_indices and neg_indices are boolean masks of shape (out_features, in_features)
    if pos_indices.any(): # Check if there are any positive contributions to avoid operating on empty selections
        wt_mat[pos_indices] = potential_pos_contrib[pos_indices]
    
    if neg_indices.any(): # Check if there are any negative contributions
        wt_mat[neg_indices] = potential_neg_contrib[neg_indices]

    result = torch.sum(wt_mat, dim=0) # Sum over output features to get relevance for input features
    
    return result
