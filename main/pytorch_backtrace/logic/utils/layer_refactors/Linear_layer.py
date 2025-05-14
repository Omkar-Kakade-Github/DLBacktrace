import numpy as np

def calculate_wt_fc(wts, inp, w, b, act):
    """
    Calculate the relevance propagation of a linear layer.
    
    Parameters:
    -----------
    wts : numpy.ndarray
        Weights for relevance calculation.
    inp : numpy.ndarray
        Input values.
    w : tensor
        Weight tensor of the layer.
    b : tensor
        Bias tensor of the layer.
    act : dict
        Activation function details.
        
    Returns:
    --------
    numpy.ndarray
        Weighted matrix for relevance propagation.
    """
    # Convert tensors to numpy arrays
    w_np = w.numpy().T
    b_np = b.numpy()
    
    # Calculate multiplication matrix efficiently
    mul_mat = np.einsum("ij,i->ji", w_np, inp)
    wt_mat = np.zeros_like(mul_mat)
    
    # Pre-compute positive and negative indices for each row
    pos_indices = mul_mat > 0
    neg_indices = mul_mat < 0
    
    # Pre-compute positive and negative sums for each row
    p_sums = np.sum(mul_mat * pos_indices, axis=1)
    n_sums = -np.sum(mul_mat * neg_indices, axis=1)
    
    # Split bias into positive and negative components
    p_bias = np.maximum(b_np, 0)
    n_bias = -np.minimum(b_np, 0)
    
    # Calculate total sum
    t_sums = p_sums + p_bias - n_sums - n_bias
    
    # Initialize aggregation weights
    p_agg_wts = np.zeros_like(p_sums)
    n_agg_wts = np.zeros_like(n_sums)
    
    # Process activation constraints
    if act["type"] == "mono":
        # Apply lower bound constraint
        if act["range"]["l"]:
            mask = t_sums < act["range"]["l"]
            p_sums[mask] = 0
            
        # Apply upper bound constraint
        if act["range"]["u"]:
            mask = t_sums > act["range"]["u"]
            n_sums[mask] = 0
            
    elif act["type"] == "non_mono":
        # Apply activation function
        func = act["func"]
        t_acts = np.vectorize(func)(t_sums)
        p_acts = np.vectorize(func)(p_sums + p_bias)
        n_acts = np.vectorize(func)(-1 * (n_sums + n_bias))
        
        # Apply range constraints
        if act["range"]["l"]:
            mask = t_sums < act["range"]["l"]
            p_sums[mask] = 0
            
        if act["range"]["u"]:
            mask = t_sums > act["range"]["u"]
            n_sums[mask] = 0
        
        # Check activation equality conditions
        mask_p_n_positive = (p_sums > 0) & (n_sums > 0)
        mask_t_eq_p = t_acts == p_acts
        mask_t_eq_n = t_acts == n_acts
        
        # Zero out based on activation equality
        p_sums[mask_p_n_positive & mask_t_eq_n] = 0
        n_sums[mask_p_n_positive & mask_t_eq_p] = 0
    
    # Calculate positive aggregation weights
    mask_p_positive = p_sums > 0
    denominators = p_sums + n_sums + p_bias + n_bias
    p_agg_wts[mask_p_positive] = (p_sums[mask_p_positive] + p_bias[mask_p_positive]) / denominators[mask_p_positive]
    p_agg_wts[mask_p_positive] *= p_sums[mask_p_positive] / (p_sums[mask_p_positive] + p_bias[mask_p_positive])
    
    # Calculate negative aggregation weights
    mask_n_positive = n_sums > 0
    n_agg_wts[mask_n_positive] = (n_sums[mask_n_positive] + n_bias[mask_n_positive]) / denominators[mask_n_positive]
    n_agg_wts[mask_n_positive] *= n_sums[mask_n_positive] / (n_sums[mask_n_positive] + n_bias[mask_n_positive])
    
    # Avoid division by zero
    p_sums_safe = np.where(p_sums == 0, 1, p_sums)
    n_sums_safe = np.where(n_sums == 0, 1, n_sums)
    
    # Apply weights to each row
    for i in range(mul_mat.shape[0]):
        # Calculate weights for positive components
        pos_mask = pos_indices[i]
        wt_mat[i, pos_mask] = (mul_mat[i, pos_mask] / p_sums_safe[i]) * wts[i] * p_agg_wts[i]
        
        # Calculate weights for negative components
        neg_mask = neg_indices[i]
        wt_mat[i, neg_mask] = (mul_mat[i, neg_mask] / n_sums_safe[i]) * wts[i] * n_agg_wts[i] * -1.0
    
    # Sum across rows to get final weighted matrix
    return np.sum(wt_mat, axis=0)
