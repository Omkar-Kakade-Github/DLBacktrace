import gc
import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided

def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-(beta * x)))
    return x * z

def np_wave(x, alpha=1.0):
    return (alpha * x * np.exp(1.0)) / (np.exp(-x) + np.exp(x))

def np_pulse(x, alpha=1.0):
    return alpha * (1 - np.tanh(x) * np.tanh(x))

def np_absolute(x, alpha=1.0):
    return alpha * x * np.tanh(x)

def np_hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0, 1)

def np_sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def np_tanh(x):
    z = np.tanh(x)
    return z.astype(np.float32)

class LSTM_forward(object):
    def __init__(
        self, num_cells, units, weights, return_sequence=False, go_backwards=False
    ):
        self.num_cells = num_cells
        self.units = units
        self.kernel = weights[0]
        self.recurrent_kernel = weights[1]
        self.bias = weights[2][1]
        self.return_sequence = return_sequence
        self.go_backwards = go_backwards
        self.recurrent_activation = torch.sigmoid()
        self.activation = torch.tanh()
        self.compute_log = {}
        for i in range(self.num_cells):
            self.compute_log[i] = {}
            self.compute_log[i]["inp"] = None
            self.compute_log[i]["x"] = None
            self.compute_log[i]["hstate"] = [None, None]
            self.compute_log[i]["cstate"] = [None, None]
            self.compute_log[i]["int_arrays"] = {}

    def compute_carry_and_output(self, x, h_tm1, c_tm1, cell_num):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        w=torch.as_tensor(self.recurrent_kernel[1], dtype=torch.float32)
        i = self.recurrent_activation(
            x_i + torch.dot(h_tm1_i, w[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f + torch.dot(h_tm1_f, w[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + torch.dot(h_tm1_c, w[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o + torch.dot(h_tm1_o, w[:, self.units * 3 :])
        )
        self.compute_log[cell_num]["int_arrays"]["i"] = i
        self.compute_log[cell_num]["int_arrays"]["f"] = f
        self.compute_log[cell_num]["int_arrays"]["c"] = c
        self.compute_log[cell_num]["int_arrays"]["o"] = o
        return c, o

    def calculate_lstm_cell_wt(self, inputs, states, cell_num, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        self.compute_log[cell_num]["inp"] = inputs
        self.compute_log[cell_num]["hstate"][0] = h_tm1
        self.compute_log[cell_num]["cstate"][0] = c_tm1
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        k_i, k_f, k_c, k_o = torch.split(self.kernel[1],self.kernel.size(1)//4,dim=1)
        x_i = torch.dot(inputs_i, k_i)
        x_f = torch.dot(inputs_f, k_f)
        x_c = torch.dot(inputs_c, k_c)
        x_o = torch.dot(inputs_o, k_o)
        b_i, b_f, b_c, b_o = torch.split(self.bias,self.bias.size(1)//4,dim=0)
        x_i = x_i + b_i
        x_f = x_f + b_f
        x_c = x_c + b_c
        x_o = x_o + b_o

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
        x = (x_i, x_f, x_c, x_o)
        h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)

        c, o = self.compute_carry_and_output(x, h_tm1, c_tm1, cell_num)
        h = o * self.activation(c)
        self.compute_log[cell_num]["x"] = x
        self.compute_log[cell_num]["hstate"][1] = h
        self.compute_log[cell_num]["cstate"][1] = c
        return h, [h, c]

    def calculate_lstm_wt(self, input_data):
        hstate = torch.tensor((1,self.units),dtype=torch.float32)
        cstate = torch.tensor((1,self.units),dtype=torch.float32)
        output = []
        for ind in range(input_data.shape[0]):
            inp = torch.tensor(
                input_data[ind, :].reshape((1, input_data.shape[1])), dtype=torch.float32
            )
            h, s = self.calculate_lstm_cell_wt(inp, [hstate, cstate], ind)
            hstate = s[0]
            cstate = s[1]
            output.append(h)
        return output

class LSTM_backtrace(object):
    def __init__(
        self, num_cells, units, weights, return_sequence=False, go_backwards=False
    ):
        self.num_cells = num_cells
        self.units = units
        self.kernel = weights[0]
        self.recurrent_kernel = weights[1]
        self.bias = weights[2]
        self.return_sequence = return_sequence
        self.go_backwards = go_backwards
        self.recurrent_activation = np_sigmoid
        self.activation = np_tanh

        self.compute_log = {}

    def lstm_wt_ful_conn(self, wts, inp, w, b, act):
        mul_mat = np.einsum("ij,i->ij", w, inp).T
        wt_mat = np.zeros(mul_mat.shape)
        for i in range(mul_mat.shape[0]):
            l1_ind1 = mul_mat[i]
            wt_ind1 = wt_mat[i]
            wt = wts[i]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            if len(b) > 0:
                if b[i] > 0:
                    pbias = b[i]
                    nbias = 0
                else:
                    pbias = 0
                    nbias = b[i] * -1
            else:
                pbias = 0
                nbias = 0
            t_sum = p_sum + pbias - n_sum - nbias
            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0
            elif act["type"] == "non_mono":
                t_act = act["func"](t_sum)
                p_act = act["func"](p_sum + pbias)
                n_act = act["func"](-1 * (n_sum + nbias))
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0
                if p_sum > 0 and n_sum > 0:
                    if t_act == p_act:
                        n_sum = 0
                    elif t_act == n_act:
                        p_sum = 0
            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1
            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        wt_mat = wt_mat.sum(axis=0)
        return wt_mat

    def calculate_wt_add(self, wts, inp=None):
        wt_mat = []
        inp_list = []
        for x in inp:
            wt_mat.append(np.zeros_like(x))
        wt_mat = np.array(wt_mat)
        inp_list = np.array(inp)
        for i in range(wt_mat.shape[1]):
            wt_ind1 = wt_mat[:, i]
            wt = wts[i]
            l1_ind1 = inp_list[:, i]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            t_sum = p_sum - n_sum
            p_agg_wt = 0
            n_agg_wt = 0
            if p_sum + n_sum > 0:
                p_agg_wt = p_sum / (p_sum + n_sum)
                n_agg_wt = n_sum / (p_sum + n_sum)
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1
            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
            wt_mat[:, i] = wt_ind1
        wt_mat = [i.reshape(wts.shape) for i in list(wt_mat)]
        return wt_mat

    def calculate_wt_multiply(self, wts, inp=None):
        wt_mat = []
        inp_list = []
        for x in inp:
            wt_mat.append(np.zeros_like(x))
        wt_mat = np.array(wt_mat)
        inp_list = np.array(inp)
        inp_prod = inp[0] * inp[1]
        inp_diff1 = np.abs(inp_prod - inp[0])
        inp_diff2 = np.abs(inp_prod - inp[1])
        inp_diff_sum = inp_diff1 + inp_diff2
        inp_wt1 = (inp_diff1 / inp_diff_sum) * wts
        inp_wt2 = (inp_diff2 / inp_diff_sum) * wts
        return [inp_wt1, inp_wt2]

    def compute_carry_and_output(self, wt_o, wt_c, h_tm1, c_tm1, x, cell_num):
        """Computes carry and output using split kernels."""
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = (h_tm1, h_tm1, h_tm1, h_tm1)
        x_i, x_f, x_c, x_o = x
        f = self.compute_log[cell_num]["int_arrays"]["f"].numpy()[0]
        i = self.compute_log[cell_num]["int_arrays"]["i"].numpy()[0]
        temp1 = np.dot(h_tm1_o, self.recurrent_kernel[1][:, self.units * 3 :]).astype(
            np.float32
        )
        wt_x_o, wt_temp1 = self.calculate_wt_add(wt_o, [x_o, temp1])
        wt_h_tm1_o = self.calculate_wt_fc(
            wt_temp1,
            h_tm1_o,
            self.recurrent_kernel[1][:, self.units * 3 :],
            [],
            {"type": None},
        )
        temp2 = f * c_tm1
        temp3_1 = np.dot(
            h_tm1_c, self.recurrent_kernel[1][:, self.units * 2 : self.units * 3]
        )
        temp3_2 = self.activation(x_c + temp3_1)
        temp3_3 = i * temp3_2
        wt_temp2, wt_temp3_3 = self.calculate_wt_add(wt_c, [temp2, temp3_3])
        wt_f, wt_c_tm1 = self.calculate_wt_multiply(wt_temp2, [f, c_tm1])
        wt_i, wt_temp3_2 = self.calculate_wt_multiply(wt_temp3_3, [i, temp3_2])
        wt_x_c, wt_temp3_1 = self.calculate_wt_add(wt_temp3_2, [x_c, temp3_1])
        wt_h_tm1_c = self.calculate_wt_fc(
            wt_temp3_1,
            h_tm1_c,
            self.recurrent_kernel[1][:, self.units * 2 : self.units * 3],
            [],
            {"type": None},
        )
        temp4 = np.dot(h_tm1_f, self.recurrent_kernel[1][:, self.units : self.units * 2])
        wt_x_f, wt_temp4 = self.calculate_wt_add(wt_f, [x_f, temp4])
        wt_h_tm1_f = self.calculate_wt_fc(
            wt_temp4,
            h_tm1_f,
            self.recurrent_kernel[1][:, self.units : self.units * 2],
            [],
            {"type": None},
        )
        temp5 = np.dot(h_tm1_i, self.recurrent_kernel[1][:, : self.units])
        wt_x_i, wt_temp5 = self.calculate_wt_add(wt_i, [x_i, temp5])
        wt_h_tm1_i = self.calculate_wt_fc(
            wt_temp5,
            h_tm1_i,
            self.recurrent_kernel[1][:, : self.units],
            [],
            {"type": None},
        )

        return (
            wt_x_i,
            wt_x_f,
            wt_x_c,
            wt_x_o,
            wt_h_tm1_i,
            wt_h_tm1_f,
            wt_h_tm1_c,
            wt_h_tm1_o,
            wt_c_tm1,
        )

    def calculate_lstm_cell_wt(self, cell_num, wts_hstate, wts_cstate):
        o = self.compute_log[cell_num]["int_arrays"]["o"].numpy()[0]
        c = self.compute_log[cell_num]["cstate"][1].numpy()[0]
        h_tm1 = self.compute_log[cell_num]["hstate"][0].numpy()[0]
        c_tm1 = self.compute_log[cell_num]["cstate"][0].numpy()[0]
        x = [i.numpy()[0] for i in self.compute_log[cell_num]["x"]]
        wt_o, wt_c = self.calculate_wt_multiply(
            wts_hstate, [o, self.activation(c)]
        )  # h = o * self.activation(c)
        wt_c = wt_c + wts_cstate
        (
            wt_x_i,
            wt_x_f,
            wt_x_c,
            wt_x_o,
            wt_h_tm1_i,
            wt_h_tm1_f,
            wt_h_tm1_c,
            wt_h_tm1_o,
            wt_c_tm1,
        ) = self.compute_carry_and_output(wt_o, wt_c, h_tm1, c_tm1, x, cell_num)
        wt_h_tm1 = wt_h_tm1_i + wt_h_tm1_f + wt_h_tm1_c + wt_h_tm1_o
        inputs = self.compute_log[cell_num]["inp"].numpy()[0]

        k_i, k_f, k_c, k_o = np.split(self.kernel[1], indices_or_sections=4, axis=1)
        b_i, b_f, b_c, b_o = np.split(self.bias[1], indices_or_sections=4, axis=0)

        wt_inputs_i = self.calculate_wt_fc(wt_x_i, inputs, k_i, b_i, {"type": None})
        wt_inputs_f = self.calculate_wt_fc(wt_x_f, inputs, k_f, b_f, {"type": None})
        wt_inputs_c = self.calculate_wt_fc(wt_x_c, inputs, k_c, b_c, {"type": None})
        wt_inputs_o = self.calculate_wt_fc(wt_x_o, inputs, k_o, b_o, {"type": None})

        wt_inputs = wt_inputs_i + wt_inputs_f + wt_inputs_c + wt_inputs_o

        return wt_inputs, wt_h_tm1, wt_c_tm1

    def calculate_lstm_wt(self, wts, compute_log):
        self.compute_log = compute_log
        output = []
        if self.return_sequence:
            temp_wts_hstate = wts[-1, :]
        else:
            temp_wts_hstate = wts
        temp_wts_cstate = np.zeros_like(self.compute_log[0]["cstate"][1].numpy()[0])
        for ind in range(len(self.compute_log) - 1, -1, -1):
            temp_wt_inp, temp_wts_hstate, temp_wts_cstate = self.calculate_lstm_cell_wt(
                ind, temp_wts_hstate, temp_wts_cstate
            )
            output.append(temp_wt_inp)
            if self.return_sequence and ind > 0:
                temp_wts_hstate = temp_wts_hstate + wts[ind - 1, :]
        output.reverse()
        return np.array(output)

def dummy_wt(wts, inp, *args):
    test_wt = np.zeros_like(inp)
    return test_wt

def calculate_wt_fc(wts, inp, w, b, act):
    """
    Optimized calculation of relevance propagation for a linear layer.
    
    Parameters:
    -----------
    wts : numpy.ndarray
        Weights for relevance calculation. Expected shape (O,) or (1,O).
    inp : numpy.ndarray
        Input values. Expected shape (I,) or (1,I).
    w : numpy.ndarray
        Weight tensor of the layer. Expected shape (O,I).
    b : numpy.ndarray
        Bias tensor of the layer. Expected shape (O,).
    act : dict
        Activation function details.
        
    Returns:
    --------
    numpy.ndarray
        Weighted matrix for relevance propagation, shape (I,).
    """
    w_np = w # Should be (O,I) as per VGG model layers
    b_np = b # Should be (O,)
    
    _wts = wts
    if isinstance(_wts, np.ndarray) and _wts.ndim == 2 and _wts.shape[0] == 1:
        _wts = _wts.squeeze(0) # Now shape (O,)
    
    inp_np = inp
    if isinstance(inp_np, np.ndarray) and inp_np.ndim == 2 and inp_np.shape[0] == 1:
        inp_np = inp_np.squeeze(0) # Now shape (I,)
     
    # Calculate contribution matrix: mul_mat[j,i] = w_np[j,i] * inp_np[i]
    # w_np is (O,I), inp_np is (I,). Broadcasting inp_np to (1,I) gives mul_mat (O,I).
    mul_mat = w_np * inp_np[np.newaxis, :] 
     
    # Pre-compute masks and values all at once
    pos_mask = mul_mat > 0 # Shape (O,I)
    neg_mask = mul_mat < 0 # Shape (O,I)
     
    # Use masks directly for efficient computation
    p_mul_mat = np.where(pos_mask, mul_mat, 0)
    n_mul_mat = np.where(neg_mask, -mul_mat, 0)  # Note the negation is applied here
     
    # Sum along axis=1 (sum over inputs i for each output j) for efficient reduction
    # If mul_mat is (O,I), axis=1 sums over I, result is (O,)
    p_sums = np.sum(p_mul_mat, axis=1) # Shape (O,)
    n_sums = np.sum(n_mul_mat, axis=1) # Shape (O,)
     
    # Efficiently split bias
    p_bias = np.maximum(b_np, 0) # Shape (O,)
    n_bias = -np.minimum(b_np, 0) # Shape (O,)
     
    # Total sums
    t_sums = p_sums + p_bias - n_sums - n_bias # Shape (O,)
     
    # Create aggregation weight arrays
    p_agg_wts = np.zeros_like(p_sums) # Shape (O,)
    n_agg_wts = np.zeros_like(n_sums) # Shape (O,)
     
    # Handle activation constraints - process only necessary elements
    if act["type"] == "mono":
        # Apply lower bound constraint if it exists
        if act["range"]["l"]:
            p_sums = np.where(t_sums < act["range"]["l"], 0, p_sums)
            
        # Apply upper bound constraint if it exists
        if act["range"]["u"]:
            n_sums = np.where(t_sums > act["range"]["u"], 0, n_sums)
            
    elif act["type"] == "non_mono":
        func = act["func"]
        
        # Only compute where needed - identify where both p_sums and n_sums are positive
        both_positive = (p_sums > 0) & (n_sums > 0)
        if np.any(both_positive):
            # Only vectorize and compute for necessary indices
            indices = np.where(both_positive)[0]
            
            # Compute activations only for relevant indices
            t_acts = np.zeros_like(t_sums)
            p_acts = np.zeros_like(p_sums)
            n_acts = np.zeros_like(n_sums)
            
            # Apply vectorized function only where needed
            vfunc = np.vectorize(func)
            t_acts[indices] = vfunc(t_sums[indices])
            p_acts[indices] = vfunc(p_sums[indices] + p_bias[indices])
            n_acts[indices] = vfunc(-1 * (n_sums[indices] + n_bias[indices]))
            
            # Apply equality conditions efficiently
            p_sums[indices] = np.where(t_acts[indices] == n_acts[indices], 0, p_sums[indices])
            n_sums[indices] = np.where(t_acts[indices] == p_acts[indices], 0, n_sums[indices])
        
        # Apply range constraints directly
        if act["range"]["l"]:
            p_sums = np.where(t_sums < act["range"]["l"], 0, p_sums)
            
        if act["range"]["u"]:
            n_sums = np.where(t_sums > act["range"]["u"], 0, n_sums)
    
    # Calculate denominators for all rows at once
    denominators = p_sums + n_sums + p_bias + n_bias # Shape (O,)
     
    # Avoid division by zero in denominators
    safe_denominators = np.where(denominators == 0, 1, denominators) # Reverted from 1e-9
     
    # Calculate aggregation weights in one operation
    mask_p_positive = p_sums > 0
    p_factor1 = (p_sums + p_bias) / safe_denominators
    p_factor2 = np.where(p_sums + p_bias != 0,
                         p_sums / (p_sums + p_bias), # Reverted from p_sums + p_bias + 1e-9
                         0)
    p_agg_wts = np.where(mask_p_positive, p_factor1 * p_factor2, 0) # Shape (O,)
     
    mask_n_positive = n_sums > 0
    n_factor1 = (n_sums + n_bias) / safe_denominators
    n_factor2 = np.where(n_sums + n_bias != 0,
                         n_sums / (n_sums + n_bias), # Reverted from n_sums + n_bias + 1e-9
                         0)
    n_agg_wts = np.where(mask_n_positive, n_factor1 * n_factor2, 0) # Shape (O,)
     
    # Safe divisors for p_sums and n_sums
    p_sums_safe = np.where(p_sums == 0, 1, p_sums) # Reverted from 1e-9
    n_sums_safe = np.where(n_sums == 0, 1, n_sums) # Reverted from 1e-9
        
    # Prepare weights matrix
    wt_mat = np.zeros_like(mul_mat) # Shape (O,I)
     
    # Pre-compute as much as possible outside the loop
    # _wts is (O,), p_agg_wts is (O,). Result is (O,)
    p_coeffs = _wts * p_agg_wts 
    n_coeffs = _wts * n_agg_wts * -1.0
     
    # Loop iterates O times (number of output neurons)
    for i in range(mul_mat.shape[0]):
        # Process positive values - use pre-computed coefficients
        pos_idx = pos_mask[i] # 1D boolean mask for inputs to output i, shape (I,)
        if np.any(pos_idx):
            # mul_mat[i, pos_idx] is 1D, (N_pos_inputs_for_output_i,)
            # p_sums_safe[i] is scalar
            # p_coeffs[i] is scalar
            wt_mat[i, pos_idx] = mul_mat[i, pos_idx] / p_sums_safe[i] * p_coeffs[i]
        
        # Process negative values - use pre-computed coefficients
        neg_idx = neg_mask[i] # 1D boolean mask for inputs to output i, shape (I,)
        if np.any(neg_idx):
            # mul_mat[i, neg_idx] is 1D, (N_neg_inputs_for_output_i,)
            # n_sums_safe[i] is scalar
            # n_coeffs[i] is scalar
            wt_mat[i, neg_idx] = mul_mat[i, neg_idx] / n_sums_safe[i] * n_coeffs[i]
     
    # Final summation
    # wt_mat is (O,I). Sum over O (axis 0) to get relevance for each input I. Result (I,)
    return np.sum(wt_mat, axis=0)

def calculate_wt_rshp(wts, inp=None):
    x = np.reshape(wts, inp.shape)
    return x

def calculate_wt_concat(wts, inp=None, axis=-1):
    wts=wts.T
    splits = [i.shape[axis] for i in inp]
    splits = np.cumsum(splits)
    if axis > 0:
        axis = axis - 1
    x = np.split(wts, indices_or_sections=splits, axis=axis)
    return x

def calculate_wt_add(wts, inp=None):
    wts=wts.T
    wt_mat = []
    inp_list = []
    expanded_wts = as_strided(
        wts,
        shape=(np.prod(wts.shape),),
        strides=(wts.strides[-1],),
        writeable=False,  # totally use this to avoid writing to memory in weird places
    )

    for x in inp:
        expanded_input = as_strided(
            x,
            shape=(np.prod(x.shape),),
            strides=(x.strides[-1],),
            writeable=False,  # totally use this to avoid writing to memory in weird places
        )
        inp_list.append(expanded_input)
        wt_mat.append(np.zeros_like(expanded_input))
    wt_mat = np.array(wt_mat)
    inp_list = np.array(inp_list)
    for i in range(wt_mat.shape[1]):
        wt_ind1 = wt_mat[:, i]
        wt = expanded_wts[i]
        l1_ind1 = inp_list[:, i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        t_sum = p_sum - n_sum
        p_agg_wt = 0
        n_agg_wt = 0
        if p_sum + n_sum > 0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1
        wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        wt_mat[:, i] = wt_ind1
    wt_mat = [i.reshape(wts.shape) for i in list(wt_mat)]
    return wt_mat

def calculate_start_wt(arg, scaler=None,thresholding=0.5,task="binary-classification"):
    if arg.ndim == 2:
        if task == "binary-classification" or task == "multi-class classification":
            x = np.argmax(arg[0])
            m = np.max(arg[0])
            y = np.zeros(arg.shape)
            if scaler:
                y[0][x] = scaler
            else:
                y[0][x] = m
        elif task == "bbox-regression":
            y = np.zeros(arg.shape)
            if scaler:
                y[0] = scaler
                num_non_zero_elements = np.count_nonzero(y)
                if num_non_zero_elements > 0:
                    y = y / num_non_zero_elements 
            else:
                m = np.max(arg[0])
                x = np.argmax(arg[0])
                y[0][x] = m
        else:
            x = np.argmax(arg[0])
            m = np.max(arg[0])
            y = np.zeros(arg.shape)
            if scaler:
                y[0][x] = scaler
            else:
                y[0][x] = m

    elif arg.ndim == 4 and task == "binary-segmentation":
        indices = np.where(arg > thresholding)
        y = np.zeros(arg.shape)
        if scaler:
            y[indices] = scaler
            num_non_zero_elements = np.count_nonzero(y)
            if num_non_zero_elements > 0:
                y = y / num_non_zero_elements 
        else:
            y[indices] = arg[indices]
            
    else:
        x = np.argmax(arg[0])
        m = np.max(arg[0])
        y = np.zeros(arg.shape)
        if scaler:
            y[0][x] = scaler
        else:
            y[0][x] = m
    return y[0]

def calculate_wt_passthru(wts):
    return wts
def calculate_wt_zero_pad(wts,inp,padding):
    wt_mat = wts[padding[0][0]:inp.shape[0]+padding[0][0],padding[1][0]:inp.shape[1]+padding[1][0],:]
    return wt_mat

def calculate_padding(kernel_size, inp_shape_WHC, padding_mode_str, strides_WH, const_val=0.0):
    """Calculates padding for a 3D tensor (W,H,C) based on PyTorch-like conventions.
    Args:
        kernel_size: Tuple (kW, kH).
        inp_shape_WHC: Tuple (W, H, C) of the input tensor (already transposed).
        padding_mode_str: String 'valid', 'same', or a tuple of explicit paddings (pad_w, pad_h).
        strides_WH: Tuple (sW, sH).
        const_val: Value for padding.
    Returns:
        Tuple: (padded_input_shape_WHC, final_paddings_for_np_pad)
          final_paddings_for_np_pad is like [[pad_W_before, pad_W_after], [pad_H_before, pad_H_after], [0,0]]
    """
    in_W, in_H, _ = inp_shape_WHC
    k_W, k_H = kernel_size
    s_W, s_H = strides_WH

    if padding_mode_str == 'valid':
        pad_W_before, pad_W_after = 0, 0
        pad_H_before, pad_H_after = 0, 0
    elif padding_mode_str == 'same':
        # For 'same' padding, output size is ceil(input_size / stride)
        # Effective input size needed: (out_size - 1) * stride + kernel_size
        # Total padding = Effective input size needed - input_size
        out_W = np.ceil(float(in_W) / float(s_W))
        out_H = np.ceil(float(in_H) / float(s_H))

        pad_W_total = int(max(0, (out_W - 1) * s_W + k_W - in_W))
        pad_H_total = int(max(0, (out_H - 1) * s_H + k_H - in_H))

        pad_W_before = pad_W_total // 2
        pad_W_after = pad_W_total - pad_W_before
        pad_H_before = pad_H_total // 2
        pad_H_after = pad_H_total - pad_H_before
    elif isinstance(padding_mode_str, (tuple, int)):
        if isinstance(padding_mode_str, int):
            pad_W_explicit = padding_mode_str
            pad_H_explicit = padding_mode_str
        else: # tuple
            pad_W_explicit, pad_H_explicit = padding_mode_str
        
        pad_W_before, pad_W_after = pad_W_explicit, pad_W_explicit
        pad_H_before, pad_H_after = pad_H_explicit, pad_H_explicit
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode_str}")

    final_paddings_for_np_pad = [
        [pad_W_before, pad_W_after],
        [pad_H_before, pad_H_after],
        [0, 0]  # No padding for channels dimension
    ]
    
    padded_shape_W = in_W + pad_W_before + pad_W_after
    padded_shape_H = in_H + pad_H_before + pad_H_after
    padded_input_shape_WHC = (padded_shape_W, padded_shape_H, inp_shape_WHC[2])

    return padded_input_shape_WHC, final_paddings_for_np_pad
    
def calculate_wt_conv_unit(patch, wts, w, b, act):
    # Ensure w and b are on CPU and detached before converting to numpy
    k = w.detach().cpu().numpy()
    bias = b.detach().cpu().numpy()
    b_ind = bias>0
    bias_pos = bias*b_ind
    b_ind = bias<0
    bias_neg = bias*b_ind*-1.0    
    # k shape: (C_out, C_in, kH, kW) -> o, i, h, w
    # patch shape: (kH, kW, C_in) -> h, w, i
    # conv_out component: k[o,i,h,w] * patch[h,w,i]
    conv_out = np.einsum("oihw,hwi->oihw",k,patch) # Result shape (C_out, C_in, kH, kW)
    
    p_ind_contrib = conv_out * (conv_out > 0) # Positive components
    p_sum = np.einsum("oihw->o", p_ind_contrib) # Sum positive contributions for each output channel, shape (C_out,)
    
    n_ind_contrib = conv_out * (conv_out < 0) # Negative components (values are negative)
    n_sum = np.einsum("oihw->o", n_ind_contrib) * -1.0 # Sum of absolute values of negative contributions, shape (C_out,)
    
    t_sum = p_sum - n_sum # Overall sum of contributions for each output channel (pre-bias)
    # The original t_sum was p_sum + n_sum where n_sum was already positive. So this is consistent.

    wt_mat = np.zeros_like(k) # Shape (C_out, C_in, kH, kW)
    p_saturate = np.ones_like(p_sum) # For simplicity, assume no saturation for now
    n_saturate = np.ones_like(n_sum) # For simplicity, assume no saturation for now

    # Simplified saturation logic for now, can be refined later if needed.
    # Original saturation logic was complex and might hide other issues.
    # We use total pre-bias activation plus bias for any checks if absolutely needed by activation type.
    actual_z_o_pre_bias = np.einsum("oihw,hwi->o", k, patch)
    total_output_activation_estimate = actual_z_o_pre_bias + bias

    if act["type"]=='mono':
        if act["range"]["l"] is not None:
            p_saturate = (total_output_activation_estimate >= act["range"]["l"])
        if act["range"]["u"] is not None:
            n_saturate = (total_output_activation_estimate <= act["range"]["u"])
    # Skipping non_mono saturation for this focused fix, it can be re-added.

    # Denominator for LRP rule (simplified)
    # Using Z_total = p_sum + bias_pos + n_sum_abs + bias_neg as a base for each output channel
    # Ensure p_saturate/n_saturate are applied if they modify effective p_sum/n_sum for denominator
    denominator_o = (p_sum * p_saturate) + (bias_pos * p_saturate) + \
                    (n_sum * n_saturate) + (bias_neg * n_saturate) + 1e-9 # Shape (C_out,)
    denominator_o[denominator_o == 0] = 1e-9 # Avoid division by zero if all terms are zero

    # Relevance fraction for each output channel
    rel_fraction_o = wts / denominator_o # Shape (C_out,)
    
    # Distribute relevance: R_input_contrib = Input_contrib * (R_output / Z_output)
    # Positive contributions pathway
    # p_ind_contrib are the positive k*patch components
    wt_mat_pos_path = p_ind_contrib * (rel_fraction_o * p_saturate)[:, np.newaxis, np.newaxis, np.newaxis]
    
    # Negative contributions pathway (LRP-epsilon often treats positive and negative contributions similarly, or uses alpha/beta)
    # If using a simple R_j/Z_j rule, sign of contribution matters.
    # n_ind_contrib are the negative k*patch components (values are negative)
    wt_mat_neg_path = n_ind_contrib * (rel_fraction_o * n_saturate)[:, np.newaxis, np.newaxis, np.newaxis]
    
    # Total relevance contribution to each k[o,i,h,w]*patch[h,w,i] component from output wts[o]
    wt_mat = wt_mat_pos_path + wt_mat_neg_path # Shape (C_out, C_in, kH, kW)
    
    # Sum these relevances over all output channels (o) to get total relevance for each input element (i,h,w) in the patch
    relevance_for_patch_ihw = np.sum(wt_mat, axis=0) # Sum over C_out (axis 0). Shape (C_in, kH, kW)
    
    # Transpose to (kH, kW, C_in) to match expected patch relevance shape
    relevance_for_patch_hwi = np.transpose(relevance_for_patch_ihw, (1, 2, 0))
    
    return relevance_for_patch_hwi


def calculate_wt_conv(wts, inp, w, b, padding_config, strides_config, act):
    """ Relevance propagation for Conv2D.
    Args:
        wts: Relevance from output. PyTorch Tensor or NumPy array. Shape (B,C_out,H_out,W_out) or (C_out,H_out,W_out).
        inp: Input activations. PyTorch Tensor or NumPy array. Shape (B,C_in,H_in,W_in) or (C_in,H_in,W_in).
        w: Weights (kernel). PyTorch tensor (C_out, C_in/groups, kH, kW).
        b: Bias. PyTorch tensor (C_out,).
        padding_config: Padding mode string ('same', 'valid') or int/tuple for explicit padding.
        strides_config: Strides tuple (sH, sW).
        act: Activation configuration.
    Returns:
        Propagated relevance to input. Shape matches squeezed inp (C_in,H_in,W_in) or batched (B,C_in,H_in,W_in).
    """
    # 1. Convert to numpy & CPU if they are tensors, Squeeze batch if present (assuming B=1 if 4D)
    if isinstance(inp, torch.Tensor):
        inp_np_choncpu = inp.cpu().detach().numpy()
    elif isinstance(inp, np.ndarray):
        inp_np_choncpu = inp
    else:
        raise TypeError(f"Input 'inp' must be a PyTorch Tensor or NumPy array, got {type(inp)}")

    if isinstance(wts, torch.Tensor):
        wts_np_choncpu = wts.cpu().detach().numpy()
    elif isinstance(wts, np.ndarray):
        wts_np_choncpu = wts
    else:
        raise TypeError(f"Input 'wts' must be a PyTorch Tensor or NumPy array, got {type(wts)}")

    # Ensure w and b are PyTorch tensors for calculate_wt_conv_unit
    # Determine device from one of the tensors if possible, else default to CPU
    device = None
    if isinstance(w, torch.Tensor):
        device = w.device
        w_torch_tensor = w
    elif isinstance(w, np.ndarray):
        w_torch_tensor = torch.as_tensor(w, dtype=torch.float32)
        # device can't be inferred from numpy, will be on CPU by default
    else:
        raise TypeError(f"Kernel 'w' must be a PyTorch Tensor or NumPy array, got {type(w)}")

    if isinstance(b, torch.Tensor):
        device = b.device if device is None else device # Prioritize w's device
        b_torch_tensor = b
    elif isinstance(b, np.ndarray):
        b_torch_tensor = torch.as_tensor(b, dtype=torch.float32)
        # device can't be inferred from numpy, will be on CPU by default
    else:
        raise TypeError(f"Bias 'b' must be a PyTorch Tensor or NumPy array, got {type(b)}")
    
    # If device is still None (e.g. w and b were numpy), try to get from inp or wts if they were tensors
    if device is None:
        if isinstance(inp, torch.Tensor): device = inp.device
        elif isinstance(wts, torch.Tensor): device = wts.device
        else: device = torch.device('cpu') # Fallback to CPU
    
    # Ensure tensors are on the same determined device before passing to unit or converting to numpy
    w_torch_tensor = w_torch_tensor.to(device)
    b_torch_tensor = b_torch_tensor.to(device)

    w_np_OICK = w_torch_tensor.cpu().detach().numpy() # (C_out, C_in/g, kH, kW)

    squeezed_inp_chw = inp_np_choncpu.squeeze(0) if inp_np_choncpu.ndim == 4 and inp_np_choncpu.shape[0] == 1 else inp_np_choncpu
    squeezed_wts_chow = wts_np_choncpu.squeeze(0) if wts_np_choncpu.ndim == 4 and wts_np_choncpu.shape[0] == 1 else wts_np_choncpu

    # 2. Transpose to WHC convention for spatial processing
    # inp: (C_in, H_in, W_in) -> (W_in, H_in, C_in)
    inp_whc = np.transpose(squeezed_inp_chw, (2, 1, 0))
    # wts: (C_out, H_out, W_out) -> (W_out, H_out, C_out)
    wts_whc = np.transpose(squeezed_wts_chow, (2, 1, 0))

    # 3. Prepare args for calculate_padding
    # w_np_OICK: (C_out, C_in/g, kH, kW)
    kernel_dims_for_padding_WH = (w_np_OICK.shape[3], w_np_OICK.shape[2]) # (kW, kH)
    # strides_config from PyTorch is (sH, sW). calculate_padding expects (sW, sH).
    strides_for_padding_WH = (strides_config[1], strides_config[0]) # (sW, sH)
    
    # inp_whc.shape is (W_in, H_in, C_in) - this is 3D as expected by calculate_padding
    _, np_paddings_whc_format = calculate_padding(
        kernel_dims_for_padding_WH, inp_whc.shape, padding_config, strides_for_padding_WH
    )
    
    input_padded_whc = np.pad(inp_whc, np_paddings_whc_format, 'constant', constant_values=0.0)
    out_ds_whc = np.zeros_like(input_padded_whc) # Accumulator (W_padded, H_padded, C_in)

    sW, sH = strides_for_padding_WH
    kW_kernel, kH_kernel = kernel_dims_for_padding_WH # Actual kernel dimensions

    # 4. Loop over output spatial dimensions and call calculate_wt_conv_unit
    for out_w_idx in range(wts_whc.shape[0]):       # Iterate W_out
        for out_h_idx in range(wts_whc.shape[1]):   # Iterate H_out
            # Determine patch boundaries in the padded input
            start_w, end_w = out_w_idx * sW, out_w_idx * sW + kW_kernel
            start_h, end_h = out_h_idx * sH, out_h_idx * sH + kH_kernel
            
            current_patch_kWkHCin_slice = input_padded_whc[start_w:end_w, start_h:end_h, :] # Shape (kW, kH, C_in)

            # Transpose patch for conv_unit if it expects (kH, kW, C_in)
            patch_for_unit_kHkWCin = np.transpose(current_patch_kWkHCin_slice, (1, 0, 2)) # Shape (kH, kW, C_in)
            
            current_relevance_at_output_pixel_Cout = wts_whc[out_w_idx, out_h_idx, :] # Shape (C_out,)
            
            # Pass original PyTorch tensors w,b to conv_unit as it might do .numpy() internally or expect tensors
            # Assuming calculate_wt_conv_unit returns relevance for the patch: (kH, kW, C_in)
            updates_kHkWCin = calculate_wt_conv_unit(
                patch_for_unit_kHkWCin, 
                current_relevance_at_output_pixel_Cout, 
                w_torch_tensor,  # Pass original torch tensor
                b_torch_tensor,  # Pass original torch tensor
                act
            )
            
            # Transpose updates back to (kW, kH, C_in) to add to the slice of out_ds_whc
            updates_kWkHCin_for_accumulation = np.transpose(updates_kHkWCin, (1, 0, 2)) # Shape (kW, kH, C_in)
            
            out_ds_whc[start_w:end_w, start_h:end_h, :] += updates_kWkHCin_for_accumulation
            
    # 5. Unpad to get relevance for original input dimensions
    unpadded_out_ds_whc = out_ds_whc[
        np_paddings_whc_format[0][0] : input_padded_whc.shape[0] - np_paddings_whc_format[0][1],
        np_paddings_whc_format[1][0] : input_padded_whc.shape[1] - np_paddings_whc_format[1][1],
        :
    ] # Shape (W_in, H_in, C_in)

    # 6. Transpose back to (C_in, H_in, W_in) convention
    final_relevance_chw = np.transpose(unpadded_out_ds_whc, (2, 1, 0))

    # 7. Add batch dimension back if original input was 4D
    if inp_np_choncpu.ndim == 4:
        final_relevance_chw = final_relevance_chw[np.newaxis, ...]
        
    return final_relevance_chw


def calculate_wt_max_unit(patch, wts, pool_size):
    pmax = np.einsum("ijk,k->ijk",np.ones_like(patch),np.max(np.max(patch,axis=0),axis=0))
    indexes = (patch-pmax)==0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0/np.einsum("mnc->c",indexes)
    indexes = np.einsum("ijk,k->ijk",indexes,indexes_norm)
    out = np.einsum("ijk,k->ijk",indexes,wts)
    return out

def calculate_wt_maxpool(wts, inp, pool_size_tpl, padding_mode_str_or_int, stride_int):
    """ Relevance propagation for MaxPool2D.
    Args:
        wts: Relevance from output. Expected shape (OutW, OutH, C) after squeeze and transpose.
        inp: Input activations. Original shape (B,C,H,W). Squeezed to (C,H,W) then transposed to (W,H,C) for this func.
        pool_size_tpl: Kernel size (kH, kW) - Note: PyTorch MaxPool2d takes (kH,kW) but internal logic might use W,H order.
                     Let's assume it's (kW, kH) to match the (W,H,C) convention of inp here.
        padding_mode_str_or_int: Padding mode string ('same', 'valid') or int for symmetric padding.
        stride_int: Stride value (int, assumed symmetric for W and H).
    Returns:
        Propagated relevance to input. Shape (W,H,C), to be transposed back before adding to all_wt.
    """
    
    # Squeeze batch dimension if present from input relevance `wts` and input activations `inp`
    squeezed_wts = wts
    if isinstance(squeezed_wts, np.ndarray) and squeezed_wts.ndim == 4 and squeezed_wts.shape[0] == 1:
        squeezed_wts = squeezed_wts.squeeze(0) # Now (C, OutH, OutW)
    
    squeezed_inp = inp
    if isinstance(squeezed_inp, np.ndarray) and squeezed_inp.ndim == 4 and squeezed_inp.shape[0] == 1:
        squeezed_inp = squeezed_inp.squeeze(0) # Now (C, InH, InW)

    # Transpose to (W,H,C) convention for processing
    # wts: (C, OutH, OutW) -> (OutW, OutH, C)
    # inp: (C, InH, InW)  -> (InW, InH, C)
    wts_whc = np.transpose(squeezed_wts, (2, 1, 0))
    inp_whc = np.transpose(squeezed_inp, (2, 1, 0))

    # Ensure pool_size and strides are tuples (kW, kH) and (sW, sH)
    if isinstance(pool_size_tpl, int):
        k_W, k_H = pool_size_tpl, pool_size_tpl
    else:
        k_H_pytorch, k_W_pytorch = pool_size_tpl # PyTorch MaxPool2d kernel_size is (kH, kW)
        k_W, k_H = k_W_pytorch, k_H_pytorch # Match our W,H order convention
        
    if isinstance(stride_int, int):
        s_W, s_H = stride_int, stride_int
    else: # Assuming tuple if not int, (sH, sW) from PyTorch
        s_H_pytorch, s_W_pytorch = stride_int
        s_W, s_H = s_W_pytorch, s_H_pytorch

    _ , np_paddings = calculate_padding((k_W, k_H), inp_whc.shape, padding_mode_str_or_int, (s_W, s_H), -np.inf)
    input_padded_whc = np.pad(inp_whc, np_paddings, 'constant', constant_values=-np.inf)
    
    out_ds_whc = np.zeros_like(input_padded_whc)

    # Iterate over the output dimensions (OutW, OutH)
    for out_w_idx in range(wts_whc.shape[0]): # Iterate OutW
        for out_h_idx in range(wts_whc.shape[1]): # Iterate OutH
            # Determine patch boundaries in the padded input
            start_w = out_w_idx * s_W
            start_h = out_h_idx * s_H
            end_w = start_w + k_W
            end_h = start_h + k_H
            
            # Slice the patch from the padded input
            current_patch_whc = input_padded_whc[start_w:end_w, start_h:end_h, :]
            
            # Get relevance for the current output pixel (all channels)
            current_wts_c = wts_whc[out_w_idx, out_h_idx, :] # Shape (C,)
            
            # Calculate relevance for the patch
            updates_whc = calculate_wt_max_unit(current_patch_whc, current_wts_c, (k_W, k_H))
            
            # Add updates to the corresponding region in out_ds
            out_ds_whc[start_w:end_w, start_h:end_h, :] += updates_whc
    
    # Unpad: Extract the original input dimensions from the padded result
    unpadded_out_ds_whc = out_ds_whc[
        np_paddings[0][0] : input_padded_whc.shape[0] - np_paddings[0][1],
        np_paddings[1][0] : input_padded_whc.shape[1] - np_paddings[1][1],
        :
    ]
    
    # Transpose back from (W,H,C) to (C,H,W) to match all_wt[key] convention before adding
    final_relevance_chw = np.transpose(unpadded_out_ds_whc, (2, 1, 0))
    
    # Add batch dimension back for consistency if original wts/inp were 4D
    if wts.ndim == 4:
        final_relevance_chw = final_relevance_chw[np.newaxis, ...]
        
    return final_relevance_chw


def calculate_wt_avg_unit(patch, wts, pool_size):
    p_ind = patch>0
    p_ind = patch*p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = patch<0
    n_ind = patch*n_ind
    n_sum = np.einsum("ijk->k",n_ind)*-1.0
    t_sum = p_sum+n_sum
    wt_mat = np.zeros_like(patch)
    p_saturate = p_sum>0
    n_saturate = n_sum>0
    t_sum[t_sum==0] = 1.0
    p_agg_wt = (1.0/(t_sum))*wts*p_saturate
    n_agg_wt = (1.0/(t_sum))*wts*n_saturate
    wt_mat = wt_mat+(p_ind*p_agg_wt)
    wt_mat = wt_mat+(n_ind*n_agg_wt*-1.0)
    return wt_mat

def calculate_wt_avgpool(wts, inp, pool_size, padding, strides):
    wts=wts.T
    inp=inp.T

    pad1 = pool_size[0]
    pad2 = pool_size[1]
    strides = (strides,strides)
    padding = (padding,padding)
    input_padded, paddings = calculate_padding(pool_size, inp, padding, strides, -np.inf)
    out_ds = np.zeros_like(input_padded)
    for ind1 in range(wts.shape[0]):
        for ind2 in range(wts.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+pool_size[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+pool_size[1])]
            # Take slice
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates = calculate_wt_avg_unit(tmp_patch, wts[ind1,ind2,:], pool_size)
            # Build tensor with "filtered" gradient
            out_ds[np.ix_(indexes[0],indexes[1])]+=updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                    paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds

def calculate_wt_gavgpool(wts, inp):
    channels = wts.shape[0]
    inp = inp.T
    wts = wts.T
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[..., c]
        temp_wt = wt_mat[..., c]
        x = inp[..., c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[x < 0] = 0
        n_mat[x > 0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat) * -1
        p_agg_wt = 0.0
        n_agg_wt = 0.0
        if p_sum + n_sum > 0.0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0.0:
            p_sum = 1.0
        if n_sum == 0.0:
            n_sum = 1.0
        temp_wt = temp_wt + ((p_mat / p_sum) * wt * p_agg_wt)
        temp_wt = temp_wt + ((n_mat / n_sum) * wt * n_agg_wt * -1.0)
        wt_mat[..., c] = temp_wt
    return wt_mat

def calculate_wt_gmaxpool_2d(wts, inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        x = inp[..., c]
        max_val = np.max(x)
        max_indexes = (x == max_val).astype(np.float32)
        max_indexes_norm = 1.0 / np.sum(max_indexes)
        max_indexes = max_indexes * max_indexes_norm
        wt_mat[..., c] = max_indexes * wt
    return wt_mat

def calculate_padding_1d(kernel_size, inp, padding, strides, const_val=0.0):
    if padding == 'valid':
        return inp, [[0, 0],[0,0]]
    elif padding == 0:
        return inp,  [[0, 0],[0,0]]
    elif isinstance(padding, int):
        inp_pad = np.pad(inp, ((padding, padding), (0,0)), 'constant', constant_values=const_val)
        return inp_pad, [[padding, padding],[0,0]]
    else:
        remainder = inp.shape[0] % strides
        if remainder == 0:
            pad_total = max(0, kernel_size - strides)
        else:
            pad_total = max(0, kernel_size - remainder)
        
        pad_left = int(np.floor(pad_total / 2.0))
        pad_right = int(np.ceil(pad_total / 2.0))
        
        inp_pad = np.pad(inp, ((pad_left, pad_right),(0,0)), 'constant', constant_values=const_val)
        return inp_pad, [[pad_left, pad_right],[0,0]]

def calculate_wt_conv_unit_1d(patch, wts, w, b, act):
    k = w.numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0
    conv_out = np.einsum("ijk,ij->ijk", k, patch)
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    n_sum = np.einsum("ijk->k",n_ind) * -1.0
    t_sum = p_sum + n_sum
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    if act["type"] == 'mono':
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = temp_ind
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate

    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat


def calculate_padding_1d_v2(kernel_size, input_length, padding, strides, dilation=1, const_val=0.0):
    """
    Calculate and apply padding to match TensorFlow Keras behavior for 'same', 'valid', and custom padding.
    
    Parameters:
        kernel_size (int): Size of the convolutional kernel.
        input_length (int): Length of the input along the spatial dimension.
        padding (str/int/tuple): Padding type. Can be:
            - 'valid': No padding.
            - 'same': Pads to maintain output length equal to input length (stride=1).
            - int: Symmetric padding on both sides.
            - tuple/list: Explicit padding [left, right].
        strides (int): Stride size of the convolution.
        dilation (int): Dilation rate for the kernel.
        const_val (float): Value used for padding. Defaults to 0.0.
    
    Returns:
        padded_length (int): Length of the input after padding.
        paddings (list): Padding applied on left and right sides.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1  # Effective size considering dilation

    if padding == 'valid':
        return input_length, [0, 0]
    elif padding == 'same':
        # Total padding required to keep output size same as input
        pad_total = max(0, (input_length - 1) * strides + effective_kernel_size - input_length)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
    elif isinstance(padding, int):
        pad_left = padding
        pad_right = padding
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        pad_left, pad_right = padding
    else:
        raise ValueError("Invalid padding. Use 'valid', 'same', an integer, or a tuple/list of two integers.")

    padded_length = input_length + pad_left + pad_right
    return padded_length, [pad_left, pad_right]


def calculate_wt_conv_unit_1d_v2(patch, wts, w, b, act):
    """
    Compute relevance for a single patch of the input tensor.

    Parameters:
        patch (ndarray): Patch of input corresponding to the receptive field of the kernel.
        wts (ndarray): Relevance values from the next layer for this patch.
        w (ndarray): Weights of the convolutional kernel.
        b (ndarray): Bias values for the convolution.
        act (dict): Activation function details. Should contain:
            - "type": Type of activation ('mono' or 'non_mono').
            - "range": Range dictionary with "l" (lower bound) and "u" (upper bound).
            - "func": Function to apply for activation.

    Returns:
        wt_mat (ndarray): Weighted relevance matrix for the patch.
    """
    kernel = w
    bias = b
    wt_mat = np.zeros_like(kernel)
    # Compute convolution output
    conv_out = np.einsum("ijk,ij->ijk", kernel, patch)
    # Separate positive and negative contributions
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    n_sum = np.einsum("ijk->k",n_ind) * -1.0
    t_sum = p_sum + n_sum
    # Handle positive and negative bias
    bias_pos = bias * (bias > 0)
    bias_neg = bias * (bias < 0) * -1.0
    # Activation handling (saturate weights if specified)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    if act["type"] == 'mono':
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = temp_ind
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind

    # Aggregate weights
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate

    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)

    return wt_mat

def calculate_wt_conv_1d(wts, inp, w, b, padding, stride, dilation, groups, act):
    """
    Perform relevance propagation for a 1D convolution layer with support for groups and dilation.

    Parameters:
        wts (ndarray): Relevance values from the next layer (shape: [output_length, output_channels]).
        inp (ndarray): Input tensor for the current layer (shape: [input_length, input_channels]).
        w (ndarray): Weights of the convolutional kernel (shape: [kernel_size, input_channels/groups, output_channels/groups]).
        b (ndarray): Bias values for the convolution (shape: [output_channels]).
        padding (str/int/tuple): Padding mode. Supports 'same', 'valid', integer, or tuple of (left, right).
        stride (int): Stride of the convolution.
        dilation (int): Dilation rate for the kernel.
        groups (int): Number of groups for grouped convolution.
        act (dict): Activation function details.

    Returns:
        out_ds (ndarray): Propagated relevance for the input tensor.
    """
    wts = wts.T
    inp = inp.T
    w = w.T    
    kernel_size = w.shape[0]
    input_length = inp.shape[0]

    # Compute and apply padding
    padded_length, paddings = calculate_padding_1d_v2(kernel_size, input_length, padding, stride, dilation)
    inp_padded = np.pad(inp, ((paddings[0], paddings[1]), (0, 0)), 'constant', constant_values=0)
    # Initialize output relevance map
    out_ds = np.zeros_like(inp_padded)

    # Handle grouped convolution
    input_channels_per_group = inp.shape[1] // groups
    output_channels_per_group = wts.shape[1] // groups

    for g in range(groups):
        input_start = g * input_channels_per_group
        input_end = (g + 1) * input_channels_per_group
        output_start = g * output_channels_per_group
        output_end = (g + 1) * output_channels_per_group

        for ind in range(wts.shape[0]):
            start_idx = ind * stride
            tmp_patch = inp_padded[start_idx:start_idx + kernel_size * dilation:dilation, input_start:input_end]
            updates = calculate_wt_conv_unit_1d_v2(tmp_patch, wts[ind, output_start:output_end], w[:, :, output_start:output_end], b[output_start:output_end], act)
            out_ds[start_idx:start_idx + kernel_size * dilation:dilation, input_start:input_end] += updates

    # Remove padding
    out_ds = out_ds[paddings[0]:(paddings[0] + input_length), :]
    return out_ds


def calculate_wt_max_unit_1d(patch, wts):
    pmax = np.max(patch, axis=0)
    indexes = (patch - pmax) == 0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0 / np.sum(indexes, axis=0)
    indexes = np.einsum("ij,j->ij", indexes, indexes_norm)
    out = np.einsum("ij,j->ij", indexes, wts)
    return out

def calculate_wt_maxpool_1d(wts, inp, pool_size, padding, stride):
    inp = inp.T
    wts = wts.T
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding, stride, -np.inf)
    out_ds = np.zeros_like(input_padded)
    stride=stride
    pool_size=pool_size
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + pool_size)
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_max_unit_1d(tmp_patch, wts[ind, :])
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds

def calculate_wt_avg_unit_1d(patch, wts):
    p_ind = patch > 0
    p_ind = patch * p_ind
    p_sum = np.sum(p_ind, axis=0)
    n_ind = patch < 0
    n_ind = patch * n_ind
    n_sum = np.sum(n_ind, axis=0) * -1.0
    t_sum = p_sum + n_sum
    wt_mat = np.zeros_like(patch)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    t_sum[t_sum == 0] = 1.0
    p_agg_wt = (1.0 / t_sum) * wts * p_saturate
    n_agg_wt = (1.0 / t_sum) * wts * n_saturate
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    return wt_mat

def calculate_wt_avgpool_1d(wts, inp, pool_size, padding, stride):
    wts = wts.T
    inp = inp.T
    stride=stride
    pool_size=pool_size
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding[0], stride[0], 0)
    out_ds = np.zeros_like(input_padded)
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride[0], ind * stride[0] + pool_size[0])
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_avg_unit_1d(tmp_patch, wts[ind, :])
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds

def calculate_wt_gavgpool_1d(wts, inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        temp_wt = wt_mat[:, c]
        x = inp[:, c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[p_mat < 0] = 0
        n_mat[n_mat > 0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat) * -1
        p_agg_wt = 0.0
        n_agg_wt = 0.0
        if p_sum + n_sum > 0.0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0.0:
            p_sum = 1.0
        if n_sum == 0.0:
            n_sum = 1.0
        temp_wt = temp_wt + ((p_mat / p_sum) * wt * p_agg_wt)
        temp_wt = temp_wt + ((n_mat / n_sum) * wt * n_agg_wt * -1.0)
        wt_mat[:, c] = temp_wt
    return wt_mat

def calculate_wt_gmaxpool_1d(wts, inp):
    wts = wts.T
    inp = inp.T
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        x = inp[:, c]
        max_val = np.max(x)
        max_indexes = (x == max_val).astype(np.float32)
        max_indexes_norm = 1.0 / np.sum(max_indexes)
        max_indexes = max_indexes * max_indexes_norm
        wt_mat[:, c] = max_indexes * wt
    return wt_mat

def calculate_output_padding_conv2d_transpose(input_shape, kernel_size, padding, strides):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides[0] + kernel_size[0],
                     (input_shape[1] - 1) * strides[1] + kernel_size[1]]
        paddings = [[0, 0], [0, 0], [0, 0]]
    elif padding == (0,0):
        out_shape = [(input_shape[0] - 1) * strides[0] + kernel_size[0],
                     (input_shape[1] - 1) * strides[1] + kernel_size[1]]
        paddings = [[0, 0], [0, 0], [0, 0]]
    elif isinstance(padding, tuple) and padding != (None, None):
        out_shape = [input_shape[0] * strides[0], input_shape[1] * strides[1]]
        pad_h = padding[0]
        pad_v = padding[1]
        paddings = [[pad_h, pad_h], [pad_v, pad_v], [0, 0]]
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides[0], input_shape[1] * strides[1]]
        pad_h = max(0, (input_shape[0] - 1) * strides[0] + kernel_size[0] - out_shape[0])
        pad_v = max(0, (input_shape[1] - 1) * strides[1] + kernel_size[1] - out_shape[1])
        paddings = [[pad_h // 2, pad_h - pad_h // 2], 
                    [pad_v // 2, pad_v - pad_v // 2], 
                    [0, 0]]
    
    return out_shape, paddings

def calculate_wt_conv2d_transpose_unit(patch, wts, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, 1, -1)
    elif patch.ndim == 2:
        patch = patch.reshape(1, *patch.shape)
    elif patch.ndim != 3:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")

    k = w.permute(0, 1, 3, 2).numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0  
    
    conv_out = np.einsum('ijkl,mnk->ijkl', k, patch)    
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    
    p_sum = np.einsum("ijkl->l", p_ind)
    n_sum = np.einsum("ijkl->l", n_ind) * -1.0
    t_sum = p_sum + n_sum
    
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    
    if act["type"] == 'mono':
        if act["range"]["l"]:
            p_saturate = t_sum > act["range"]["l"]
        if act["range"]["u"]:
            n_saturate = t_sum < act["range"]["u"]
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate
    
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_wt_conv2d_transpose(wts, inp, w, b, padding, strides, act):
    wts = wts.T
    inp = inp.T
    w = w.T
    out_shape, paddings = calculate_output_padding_conv2d_transpose(inp.shape, w.shape, padding, strides)
    out_ds = np.zeros(out_shape + [w.shape[3]])
    
    for ind1 in range(inp.shape[0]):
        for ind2 in range(inp.shape[1]):
            out_ind1 = ind1 * strides[0]
            out_ind2 = ind2 * strides[1]
            tmp_patch = inp[ind1, ind2, :]
            updates = calculate_wt_conv2d_transpose_unit(tmp_patch, wts[ind1, ind2, :], w, b, act)
            end_ind1 = min(out_ind1 + w.shape[0], out_shape[0])
            end_ind2 = min(out_ind2 + w.shape[1], out_shape[1])
            valid_updates = updates[:end_ind1 - out_ind1, :end_ind2 - out_ind2, :]
            out_ds[out_ind1:end_ind1, out_ind2:end_ind2, :] += valid_updates
    
    if padding == 'same':
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds.shape[0], (i+1) * strides[0])
                end_j = min(out_ds.shape[1], (j+1) * strides[1])
                relevant_area = out_ds[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds[i, j, :] = np.sum(relevant_area, axis=(0, 1))
        out_ds = adjusted_out_ds
    elif isinstance(padding, tuple) and padding != (None, None):
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds.shape[0], (i+1) * strides[0])
                end_j = min(out_ds.shape[1], (j+1) * strides[1])
                relevant_area = out_ds[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds[i, j, :] = np.sum(relevant_area, axis=(0, 1))
        out_ds = adjusted_out_ds
    else:
        out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0]),
                        paddings[1][0]:(paddings[1][0] + inp.shape[1]), :]
    
    return out_ds


def calculate_output_padding_conv1d_transpose(input_shape, kernel_size, padding, strides,dilation):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        paddings = [[0, 0], [0, 0]]
    elif padding == 0:
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        paddings = [[0, 0], [0, 0]]
    elif isinstance(padding, int):
        out_shape = [input_shape[0] * strides]
        pad_v = (dilation * (kernel_size[0] - 1)) - padding
        out_shape = [input_shape[0] * strides + pad_v]
        paddings = [[pad_v, pad_v], 
                    [0, 0]]
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides]
        pad_h = max(0, (input_shape[0] - 1) * strides + kernel_size[0] - out_shape[0])
        paddings = [[pad_h // 2, pad_h // 2], 
                    [0, 0]]
    
    return out_shape, paddings

def calculate_wt_conv1d_transpose_unit(patch, wts, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, -1)
    elif patch.ndim != 2:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")
    
    k = w.permute(0, 2, 1).numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0  
    conv_out = np.einsum('ijk,mj->ijk', k, patch)
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    
    p_sum = np.einsum("ijl->l", p_ind)
    n_sum = np.einsum("ijl->l", n_ind) * -1.0
    t_sum = p_sum + n_sum
    
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    
    if act["type"] == 'mono':
        if act["range"]["l"]:
            p_saturate = t_sum > act["range"]["l"]
        if act["range"]["u"]:
            n_saturate = t_sum < act["range"]["u"]
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_wt_conv1d_transpose(wts, inp, w, b, padding, strides, dilation, act):
    wts = wts.T
    inp = inp.T
    w = w.T
    out_shape, paddings = calculate_output_padding_conv1d_transpose(inp.shape, w.shape, padding, strides, dilation)
    out_ds = np.zeros(out_shape + [w.shape[2]])

    for ind in range(inp.shape[0]):
        out_ind = ind * strides
        tmp_patch = inp[ind, :]
        updates = calculate_wt_conv1d_transpose_unit(tmp_patch, wts[ind, :], w, b, act)
        end_ind = min(out_ind + w.shape[0], out_shape[0])
        valid_updates = updates[:end_ind - out_ind, :]
        out_ds[out_ind:end_ind, :] += valid_updates
    
    if padding == 'same':
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds.shape[0], (i + 1) * strides)
            relevant_area = out_ds[start_i:end_i, :]
            adjusted_out_ds[i, :] = np.sum(relevant_area, axis=0)
        out_ds = adjusted_out_ds
    elif padding == 0:
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds.shape[0], (i + 1) * strides)
            relevant_area = out_ds[start_i:end_i, :]
            adjusted_out_ds[i, :] = np.sum(relevant_area, axis=0)
        out_ds = adjusted_out_ds
    else:
        out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0]), :]
    return out_ds


####################################################################
###################    Encoder Model    ####################
####################################################################
def stabilize(matrix, epsilon=1e-6):
    return matrix + epsilon * np.sign(matrix)


def calculate_wt_residual(wts, inp=None):
    wt_mat = []
    inp_list = []
    expanded_wts = as_strided(
        wts,
        shape=(np.prod(wts.shape),),
        strides=(wts.strides[-1],),
        writeable=False,  # totally use this to avoid writing to memory in weird places
    )

    for x in inp:
        expanded_input = as_strided(
            x,
            shape=(np.prod(x.shape),),
            strides=(x.strides[-1],),
            writeable=False,  # totally use this to avoid writing to memory in weird places
        )
        inp_list.append(expanded_input)
        wt_mat.append(np.zeros_like(expanded_input))
    wt_mat = np.array(wt_mat)
    inp_list = np.array(inp_list)
    for i in range(wt_mat.shape[1]):
        wt_ind1 = wt_mat[:, i]
        wt = expanded_wts[i]
        l1_ind1 = inp_list[:, i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        t_sum = p_sum - n_sum
        p_agg_wt = 0
        n_agg_wt = 0
        if p_sum + n_sum > 0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1
        wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        wt_mat[:, i] = wt_ind1
    wt_mat = [i.reshape(wts.shape) for i in list(wt_mat)]
    return wt_mat


def calculate_relevance_V(wts, value_output, w):
    wt_mat_V = np.zeros(value_output.shape)
    
    if 'b_v' in w:
        bias_v = w['b_v']
    else:
        bias_v = 0

    for i in range(wts.shape[0]):
        for j in range(wts.shape[1]):
            l1_ind1 = value_output
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            
            if bias_v[i] > 0:
                pbias = bias_v[i]
                nbias = 0
            else:
                pbias = 0
                nbias = bias_v[i] * -1

            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0

            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_mat_V[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat_V[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return wt_mat_V


def calculate_relevance_QK(wts, QK_output, w):
    wt_mat_QK = np.zeros(QK_output.shape)
    
    # Check if 'b_q' and 'b_k' exist in the weights, default to 0 if not
    b_q = w['b_q'] if 'b_q' in w else 0
    b_k = w['b_k'] if 'b_k' in w else 0

    for i in range(wts.shape[0]):
        for j in range(wts.shape[1]):
            l1_ind1 = QK_output
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            if b_q[i] > 0 and b_k[i] > 0:
                pbias = b_q[i] + b_k[i]
                nbias = 0
            elif b_q[i] > 0 and b_k[i] < 0:
                pbias = b_q[i]
                nbias = b_k[i] * -1
            elif b_q[i] < 0 and b_k[i] > 0:
                pbias = b_k[i]
                nbias = b_q[i] * -1
            else:
                pbias = 0
                nbias = b_q[i] + b_k[i]
                nbias *= -1

            t_sum = p_sum + pbias - n_sum - nbias

            # This layer has a softmax activation function
            act = {
                "name": "softmax",
                "range": {"l": -1, "u": 2},
                "type": "mono",
                "func": None,
            }

            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0

            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0

            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_mat_QK[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat_QK[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return  wt_mat_QK


def calculate_wt_attention_output_projection(wts, proj_output, w):
    wt_mat_proj_output = np.zeros(proj_output.shape)
    
    if 'b_d' in w:
        bias_d = w['b_d']
    else:
        bias_d = 0

    for i in range(wts.shape[0]):
        for j in range(wts.shape[1]):
            l1_ind1 = proj_output
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            
            if bias_d[i] > 0:
                pbias = bias_d[i]
                nbias = 0
            else:
                pbias = 0
                nbias = bias_d[i] * -1

            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0
                
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_mat_proj_output[p_ind] += (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_mat_proj_output[n_ind] += (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    return wt_mat_proj_output


def calculate_wt_self_attention(wts, inp, w, config):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_q', 'W_k', 'W_v', 'W_o']

    Outputs:
        Step-1: outputs = torch.matmul(input_a, input_b)
        Step-2: outputs = F.softmax(inputs, dim=dim, dtype=dtype)
        Step-3: outputs = input_a * input_b
    '''
    # print(f"inp: {inp.shape}, wts: {wts.shape}")   # (1, 512)
    # print(f"w['W_q']: {w['W_q'].shape}, w['W_k']: {w['W_k'].shape}, w['W_v']: {w['W_v'].shape}")

    query_output = np.einsum('ij,kj->ik', inp, w['W_q'])
    key_output = np.einsum('ij,kj->ik', inp, w['W_k'])
    value_output = np.einsum('ij,kj->ik', inp, w['W_v'])

    # --------------- Reshape for Multi-Head Attention ----------------------
    num_heads = getattr(config, 'num_attention_heads', getattr(config, 'num_heads', None))     # will work for BERT as well as T5/ Llama
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', None))             # will work for BERT as well as T5/Llama
    if hasattr(config, 'num_key_value_heads'):
        num_key_value_heads = config.num_key_value_heads
    else:
        num_key_value_heads = num_heads
    head_dim = hidden_size // num_heads  # dimension of each attention head

    query_states = np.einsum('thd->htd', query_output.reshape(query_output.shape[0], num_heads, head_dim))  # (num_heads, num_tokens, head_dim)
    key_states = np.einsum('thd->htd', key_output.reshape(key_output.shape[0], num_key_value_heads, head_dim))  # (num_key_value_heads, num_tokens, head_dim)
    value_states = np.einsum('thd->htd', value_output.reshape(value_output.shape[0], num_key_value_heads, head_dim))  # (num_key_value_heads, num_tokens, head_dim)

    # calculate how many times we need to repeat the key/value heads
    n_rep = num_heads // num_key_value_heads
    key_states = np.repeat(key_states, n_rep, axis=0)
    value_states = np.repeat(value_states, n_rep, axis=0)

    QK_output = np.einsum('hqd,hkd->hqk', query_states, key_states)    # (num_heads, num_tokens, num_tokens)
    attn_weights = QK_output / np.sqrt(head_dim)

    # Apply softmax along the last dimension (softmax over key dimension)
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))  # Numerically stable softmax
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # Weighted sum of values (num_heads, num_tokens, head_dim)
    attn_output = np.einsum('hqk,hkl->hql', attn_weights, value_states)

    transposed_attn_output = np.einsum('hqd->qhd', attn_output)
    reshaped_attn_output = transposed_attn_output.reshape(transposed_attn_output.shape[0], num_heads * head_dim)

    # Perform final linear projection (num_tokens, hidden_size)
    final_output = np.einsum('qd,dh->qh', reshaped_attn_output, w['W_d'])

    # ------------- Relevance calculation for Final Linear Projection -------------
    wt_mat_attn_proj = calculate_wt_attention_output_projection(wts, final_output, w)

    # --------------- Relevance Calculation for Step-3 -----------------------
    # divide the relevance among `attn_weights` and `value_states`
    wt_mat_attn_proj = wt_mat_attn_proj.reshape(-1, num_heads, head_dim)
    wt_mat_attn_proj = np.einsum('qhd->hqd', wt_mat_attn_proj)

    stabilized_attn_output = stabilize(attn_output * 2)
    norm_wt_mat_attn_proj = wt_mat_attn_proj / stabilized_attn_output
    relevance_QK = np.einsum('htd,hbd->htb', norm_wt_mat_attn_proj, value_states) * attn_weights
    relevance_V = np.einsum('hdt,hdb->htb', attn_weights, norm_wt_mat_attn_proj)  * value_states

    # --------------- Relevance Calculation for V --------------------------------
    relevance_V = np.einsum('hqd->qhd', relevance_V)
    relevance_V = relevance_V.reshape(-1, num_heads * head_dim)
    wt_mat_V = calculate_relevance_V(relevance_V, value_states, w)

    # --------------- Transformed Relevance QK ----------------------------------
    relevance_QK = np.einsum('hqd->qhd', relevance_QK)
    relevance_QK = relevance_QK.reshape(-1, relevance_QK.shape[1] * relevance_QK.shape[2])
    wt_mat_QK = calculate_relevance_QK(relevance_QK, QK_output, w)

    # --------------- Relevance Calculation for K and Q --------------------------------
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output
    wt_mat_Q = np.einsum('htd,hdb->htb', norm_wt_mat_QK, key_states) * query_states
    wt_mat_K = np.einsum('htd,htb->hbd', query_states, norm_wt_mat_QK) * key_states

    wt_mat = wt_mat_V + wt_mat_K + wt_mat_Q

    # Reshape wt_mat
    wt_mat = np.einsum('htd->thd', wt_mat)
    wt_mat = wt_mat.reshape(wt_mat.shape[0], wt_mat.shape[1] * wt_mat.shape[2])  # reshaped_array = array.reshape(8, 32 * 128)

    return wt_mat


def calculate_wt_feed_forward(wts, inp, w):
    intermediate_output = np.einsum('ij,jk->ik', inp, w['W_int'].T)
    feed_forward_output = np.einsum('ij,jk->ik', intermediate_output, w['W_out'].T)

    relevance_input = np.zeros(inp.shape)
    relevance_out = np.zeros(intermediate_output.shape)

    # Relevance propagation for 2nd layer
    for i in range(wts.shape[0]):
        R2 = wts[i]
        contribution_matrix2 = np.einsum('ij,j->ij', w['W_out'], intermediate_output[i])
        wt_mat2 = np.zeros(contribution_matrix2.shape)
        
        bias_out = w['b_out'] if 'b_out' in w else 0
        
        for j in range(contribution_matrix2.shape[0]):
            l1_ind1 = contribution_matrix2[j]
            wt_ind1 = wt_mat2[j]
            wt = R2[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            
            # Handle positive and negative bias contributions
            if bias_out[i] > 0:
                pbias = bias_out[i]
                nbias = 0
            else:
                pbias = 0
                nbias = -bias_out[i]

            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0

            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_out[i] = wt_mat2.sum(axis=0)

    # Relevance propagation for 1st layer
    for i in range(relevance_out.shape[0]):
        R1 = relevance_out[i]
        contribution_matrix1 = np.einsum('ij,j->ij', w['W_int'], inp[i])
        wt_mat1 = np.zeros(contribution_matrix1.shape)
        
        # Check if bias 'b_int' exists, default to 0 if not
        bias_int = w['b_int'] if 'b_int' in w else 0

        for j in range(contribution_matrix1.shape[0]):
            l1_ind1 = contribution_matrix1[j]
            wt_ind1 = wt_mat1[j]
            wt = R1[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            # Handle positive and negative bias
            if bias_int[i] > 0:
                pbias = bias_int[i]
                nbias = 0
            else:
                pbias = 0
                nbias = -bias_int[i]

            t_sum = p_sum + pbias - n_sum - nbias

            # This layer has a ReLU activation function
            act = {
                "name": "relu",
                "range": {"l": 0, "u": None},
                "type": "mono",
                "func": None,
            }

            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0

            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0

            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_input[i] = wt_mat1.sum(axis=0)

    return relevance_input


def calculate_wt_classifier(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_cls', 'b_cls']
    '''
    mul_mat = np.einsum("ij, i->ij", w['W_cls'].T, inp).T
    wt_mat = np.zeros(mul_mat.shape)

    for i in range(mul_mat.shape[0]):
        l1_ind1 = mul_mat[i]
        wt_ind1 = wt_mat[i]
        wt = wts[i]

        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1

        if w['b_cls'][i] > 0:
            pbias = w['b_cls'][i]
            nbias = 0
        else:
            pbias = 0
            nbias = w['b_cls'][i]

        t_sum = p_sum + pbias - n_sum - nbias

        # This layer has a softmax activation function
        act = {
            "name": "softmax",
            "range": {"l": -1, "u": 2},
            "type": "mono",
            "func": None,
        }

        if act["type"] == "mono":
            if act["range"]["l"]:
                if t_sum < act["range"]["l"]:
                    p_sum = 0
            if act["range"]["u"]:
                if t_sum > act["range"]["u"]:
                    n_sum = 0

        if p_sum > 0:
            p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
            p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
        else:
            p_agg_wt = 0
        if n_sum > 0:
            n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
            n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
        else:
            n_agg_wt = 0

        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1

        wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    wt_mat = wt_mat.sum(axis=0)
    return wt_mat


def calculate_wt_pooler(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_p', 'b_p']
    '''
    relevance_inp = np.zeros(inp.shape)

    for i in range(inp.shape[0]):
        # Compute contribution matrix
        contribution_matrix = np.einsum('ij,j->ij', w['W_p'], inp[i])
        wt_mat = np.zeros(contribution_matrix.shape)

        # Iterate over each unit
        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt_ind1 = wt_mat[j]
            wt = wts[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            # Calculate biases
            pbias = max(w['b_p'][j], 0)
            nbias = min(w['b_p'][j], 0) * -1

            t_sum = p_sum + pbias - n_sum - nbias

            # This layer has a tanh activation function
            act = {
                "name": "tanh",
                "range": {"l": -2, "u": 2},
                "type": "mono",
                "func": None
            }

            # Apply activation function constraints
            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0

            # Aggregate weights based on positive and negative contributions
            p_agg_wt = 0
            n_agg_wt = 0
            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt *= (p_sum / (p_sum + pbias))

            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt *= (n_sum / (n_sum + nbias))

            # Prevent division by zero
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            # Update weight matrix
            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        # Calculate relevance for each token
        relevance_inp[i] = wt_mat.sum(axis=0)

    relevance_inp *= (np.sum(wts) / np.sum(relevance_inp))
    return relevance_inp


####################################################################
###################    Encoder-Decoder Model    ####################
####################################################################

def calculate_enc_dec_start_wt(arg, indices):
    y = np.zeros(arg.shape, dtype=np.float64)
    value = 1 / arg.shape[0]

    for i in range(arg.shape[0]):
        y[i][indices[i]] = value

    return y


def calculate_wt_lm_head(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_lm_head']
    '''
    relevance_input = np.zeros(inp.shape)

    for i in range(wts.shape[0]):
        R = wts[i]
        contribution_matrix = np.einsum('ij,j->ij', w['W_lm_head'], inp[i])
        wt_mat = np.zeros(contribution_matrix.shape)

        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt_ind1 = wt_mat[j]
            wt = R[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            if p_sum > 0:
                p_agg_wt = p_sum / (p_sum + n_sum)
            else:
                p_agg_wt = 0

            if n_sum > 0:
                n_agg_wt = n_sum / (p_sum + n_sum)
            else:
                n_agg_wt = 0

            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_input[i] = wt_mat.sum(axis=0)

    return relevance_input


def calculate_wt_cross_attention(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_q', 'W_k', 'W_v', 'W_o']
        inputs: dict_keys(['query', 'key', 'value'])

    Outputs:
        Step-1: outputs = torch.matmul(input_a, input_b)
        Step-2: outputs = F.softmax(inputs, dim=dim, dtype=dtype)
        Step-3: outputs = input_a * input_b
    '''
    k_v_inp, q_inp = inp
    query_output = np.einsum('ij,kj->ik', q_inp, w['W_q'])
    key_output = np.einsum('ij,kj->ik', k_v_inp, w['W_k'])
    value_output = np.einsum('ij,kj->ik', k_v_inp, w['W_v'])

    # --------------- Relevance Calculation for Step-3 -----------------------
    relevance_V = wts / 2
    relevance_QK = wts / 2

    # --------------- Relevance Calculation for V --------------------------------
    wt_mat_V = calculate_relevance_V(relevance_V, value_output)

    # --------------- Transformed Relevance QK ----------------------------------
    QK_output = np.einsum('ij,kj->ik', query_output, key_output)
    wt_mat_QK = calculate_relevance_QK(relevance_QK, QK_output)

    # --------------- Relevance Calculation for K and Q --------------------------------
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output
    wt_mat_Q = np.einsum('ij,jk->ik', norm_wt_mat_QK, key_output) * query_output
    wt_mat_K = np.einsum('ij,ik->kj', query_output, norm_wt_mat_QK) * key_output

    wt_mat_KV = wt_mat_V + wt_mat_K
    wt_mat = [wt_mat_KV, wt_mat_Q]
    return wt_mat
