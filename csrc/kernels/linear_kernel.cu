// DLBacktrace/csrc/kernels/linear_kernel.cu

// TODO: Include necessary CUDA headers like <cuda_runtime.h> if not handled by PyTorch's build system.
// TODO: Implement device-side activation functions if non-monotonic path is fully developed.
// Example: __device__ float device_swish(float x, float beta) { return x * (1.0f / (1.0f + expf(-beta * x))); }

__global__ void linear_backtrace_kernel(
    const float* __restrict__ output_relevance_from_next_layer, // wts in Python, shape (out_features)
    const float* __restrict__ input_activations_to_this_layer,  // inp in Python, shape (in_features)
    const float* __restrict__ weights,                          // FC layer weights, shape (out_features, in_features), row-major
    const float* __restrict__ biases,                           // FC layer biases, shape (out_features)
    float* __restrict__ input_relevance_for_previous_layer,   // Result, shape (in_features)
    int in_features,
    int out_features,
    bool act_is_mono,
    bool has_act_lower_bound,
    float act_lower_bound,
    bool has_act_upper_bound,
    float act_upper_bound
    // int act_func_type // Placeholder for specific non-monotonic function
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Current input feature index this thread is responsible for

    if (j < in_features) {
        float accumulated_relevance_for_input_j = 0.0f;

        for (int i = 0; i < out_features; ++i) { // Iterate over each output neuron 'i'
            float p_sum_local = 0.0f; 
            float n_sum_local = 0.0f; 

            for (int k = 0; k < in_features; ++k) {
                float weighted_input_k_to_i = input_activations_to_this_layer[k] * weights[i * in_features + k];
                if (weighted_input_k_to_i > 0) {
                    p_sum_local += weighted_input_k_to_i;
                } else {
                    n_sum_local -= weighted_input_k_to_i; // add absolute value
                }
            }

            float pbias_local = (biases[i] > 0) ? biases[i] : 0.0f;
            float nbias_local = (biases[i] < 0) ? -biases[i] : 0.0f;

            float t_sum_pre_activation = p_sum_local + pbias_local - n_sum_local - nbias_local;

            float p_sum_mod = p_sum_local; // Will be modified by activation logic
            float n_sum_mod = n_sum_local; // Will be modified by activation logic

            // Activation Handling (mirroring Python logic)
            if (act_is_mono) {
                if (has_act_lower_bound && t_sum_pre_activation < act_lower_bound) {
                    p_sum_mod = 0.0f;
                }
                if (has_act_upper_bound && t_sum_pre_activation > act_upper_bound) {
                    n_sum_mod = 0.0f;
                }
            } else { // Non-monotonic
                // TODO: Implement full non-monotonic logic based on act["func"]
                // This is a placeholder for the range checks similar to monotonic
                // The part comparing t_act, p_act, n_act needs specific device functions
                if (has_act_lower_bound && t_sum_pre_activation < act_lower_bound) {
                    p_sum_mod = 0.0f;
                }
                if (has_act_upper_bound && t_sum_pre_activation > act_upper_bound) {
                    n_sum_mod = 0.0f;
                }
                // Example for Swish (if act_func_type indicated Swish and device_swish was defined):
                // float t_act = device_swish(t_sum_pre_activation, 0.75f);
                // float p_act_val = device_swish(p_sum_local + pbias_local, 0.75f);
                // float n_act_val = device_swish(-(n_sum_local + nbias_local), 0.75f);
                // if (p_sum_mod > 0.0f && n_sum_mod > 0.0f) {
                //    if (abs(t_act - p_act_val) < 1e-6f) n_sum_mod = 0.0f; // Using a small tolerance for float comparison
                //    else if (abs(t_act - n_act_val) < 1e-6f) p_sum_mod = 0.0f;
                // }
            }

            // Calculate Aggregate Weights for Positive and Negative Parts
            // These use p_sum_mod and n_sum_mod directly from activation stage.
            // Division by zero here will lead to C++/CUDA's standard float inf/nan.
            float p_agg_wt = 0.0f;
            float n_agg_wt = 0.0f;

            float total_denominator_for_agg = p_sum_mod + n_sum_mod + pbias_local + nbias_local;

            if (p_sum_mod > 0.0f) {
                float term1_num = p_sum_mod + pbias_local;
                float term1 = term1_num / total_denominator_for_agg;
                
                float term2_num = p_sum_mod;
                float term2_den = p_sum_mod + pbias_local;
                float term2 = term2_num / term2_den; // This can be inf/nan if term2_den is 0
                p_agg_wt = term1 * term2;
            }

            if (n_sum_mod > 0.0f) {
                float term1_num = n_sum_mod + nbias_local;
                float term1 = term1_num / total_denominator_for_agg;

                float term2_num = n_sum_mod;
                float term2_den = n_sum_mod + nbias_local;
                float term2 = term2_num / term2_den; // This can be inf/nan if term2_den is 0
                n_agg_wt = term1 * term2;
            }
            
            // Stabilize denominators for the final relevance distribution step, as in Python
            float p_sum_stable = (p_sum_mod == 0.0f) ? 1.0f : p_sum_mod;
            float n_sum_stable = (n_sum_mod == 0.0f) ? 1.0f : n_sum_mod;

            // Distribute Relevance from output neuron 'i' to current input neuron 'j'
            float current_input_j_contribution_to_output_i =
                input_activations_to_this_layer[j] * weights[i * in_features + j];
            
            float relevance_component_from_output_i = 0.0f;

            if (current_input_j_contribution_to_output_i > 0.0f) {
                // Python: (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
                relevance_component_from_output_i = (current_input_j_contribution_to_output_i / p_sum_stable) *
                                                    output_relevance_from_next_layer[i] * p_agg_wt;
            } else if (current_input_j_contribution_to_output_i < 0.0f) {
                // Python: (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
                relevance_component_from_output_i = (current_input_j_contribution_to_output_i / n_sum_stable) *
                                                    output_relevance_from_next_layer[i] * n_agg_wt * -1.0f;
            }
            
            accumulated_relevance_for_input_j += relevance_component_from_output_i;
        }
        input_relevance_for_previous_layer[j] = accumulated_relevance_for_input_j;
    }
}   
