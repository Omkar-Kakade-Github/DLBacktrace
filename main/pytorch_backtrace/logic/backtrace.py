import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from main.pytorch_backtrace.logic.utils import contrast as UC
from main.pytorch_backtrace.logic.utils.layer_refactors import vgg_layers as UP_REFACTORED
from main.pytorch_backtrace.logic.utils import prop as UP_ORIGINAL
from main.pytorch_backtrace.logic.config import activation_master
from main.pytorch_backtrace.logic.utils import helper as HP
from main.pytorch_backtrace.logic.utils import encoder as EN
from main.pytorch_backtrace.logic.utils import encoder_decoder as ED

class Backtrace(object):
    """
    This is the constructor method for the Backtrace class. It initializes an instance of the class.
    It takes two optional parameters: model (a neural network model) and activation_dict (a dictionary that maps layer names to activation functions).
    """

    def __init__(self, model=None, activation_dict={}, model_type=None):
        if model_type == 'encoder':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure for encoder model
            self.model_resource = EN.build_encoder_tree(model)
            # create a layer stack for encoder model
            self.create_layer_stack()
            # extract the encoder model weights
            self.model_weights = EN.extract_encoder_weights(model)
            # # calculate the output of each submodule of the encoder model
            # self.all_out_model = EN.create_encoder_output(model)
            self.activation_dict = None
            
        elif model_type == 'encoder_decoder':
            self.model = model
            self.model_type = model_type
            # create a tree-like structure and layer_stack for encoder-decoder model
            self.model_resource, self.layer_stack = ED.build_enc_dec_tree(model)
            # extract the encoder-decoder model weights
            self.model_weights = ED.extract_encoder_decoder_weights(model)  
            # # calculate the output of each submodule of the encoder-decoder model
            # self.all_out_model = ED.calculate_encoder_decoder_output(model)
            self.activation_dict = None
        
        else:
            self.model_type = model_type
            # create a tree-like structure that represents the layers of the neural network model
            self.create_tree(model)
            # create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
            self.create_model_output(model)
            # create a new model (an instance of tf.keras.Model) that produces the output of each layer in the neural network.
            self.create_every_model_output(model)
            # create a layer stack that defines the order in which layers should be processed during backpropagation.
            self.create_layer_stack()
            # checks if the model is sequential or not. If it's sequential, it adds the input layer to the layer stack.
            # identity

            inp_name = 'identity'
            self.layer_stack.append(inp_name)
            self.model_resource[1][inp_name] = {}
            self.model_resource[1][inp_name]["name"] = inp_name
            self.model_resource[1][inp_name]["type"] = "input"
            self.model_resource[1][inp_name]["parent"] = []
            self.model_resource[1][inp_name]["child"] = None
            self.model_resource[3].append(inp_name)
            self.sequential = True
            try:
                # calls the build_activation_dict method to build a dictionary that maps layer names to activation functions.
                # If that fails, it creates a temporary dictionary with default activation functions.
                if len(activation_dict) == 0:
                    self.build_activation_dict(model)
                else:
                    self.activation_dict = activation_dict

            except Exception as e:
                print(e)
                temp_dict = {}
                for l in model.layers:
                    temp_dict[l.name] = activation_master["None"]
                self.activation_dict = temp_dict 

    def build_activation_dict(self, model):
        model_resource = self.model_resource
        layer_list = list(model_resource[0].keys())
        activation_dict = {}
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']  # You can add more activation functions
        for l in layer_list:
            activation_found = False
            try:  # could be activation for that layer
                for activation in activation_functions:
                    if activation in l.split('/')[1]:
                        activation_dict[l.split('/')[0]] = activation
                        activation_found = True
            except:
                activation_dict[l] = 'None'
        # activation_master :
        for key, value in activation_dict.items():
            activation_dict[key] = activation_master.get(value)
        self.activation_dict = activation_dict

    def create_tree(self, model):
        # create new layers same as tf version
        layers = list(model.named_children())
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        layer_sequence = []
        for i in range(len(layers) - 1):
            current_layer, current_layer_obj = layers[i]
            next_layer, next_layer_obj = layers[i + 1]
            current_layer_name = current_layer
            next_layer_name = next_layer

            next_layer_type = next_layer_name.lower()
            if any(af in next_layer_type for af in activation_functions):
                layer_sequence.append((f"{current_layer_name}/{next_layer_name}", current_layer_obj))
                i += 1
            else:
                if any(af in current_layer_name for af in activation_functions) is False:
                    layer_sequence.append((current_layer_name, current_layer_obj))
        # creating model_resource variable
        layer_sequence
        ltree = {}
        layer_tree = {}
        inputs = []
        outputs = []
        intermediates = []
        prev_layer_id = None
        num_layers = len(layer_sequence)
        for i, (layer_name, layer) in enumerate(layer_sequence):
            layer_id = layer_name
            ltree[layer_id] = {}
            layer_tree[layer_id] = layer
            layer_type = layer.__class__.__name__
            ltree[layer_id]["name"] = layer_id.split("/")[0]
            ltree[layer_id]["class"] = layer_type
            if i < num_layers - 1:
                ltree[layer_id]["type"] = "intermediate"
                intermediates.append(layer_id)
            else:
                ltree[layer_id]["type"] = "output"
                outputs.append(layer_id)
            if prev_layer_id is not None:
                ltree[layer_id]["child"] = [prev_layer_id]
                ltree[prev_layer_id]["parent"] = [layer_id]
            prev_layer_id = layer_id
        # Set child of the last layer as an empty list
        if prev_layer_id is not None:
            ltree[prev_layer_id]["parent"] = []
        layer_tree.pop('identity')
        ltree.pop('identity')
        self.model_resource = (layer_tree, ltree, outputs, inputs)

    def create_layer_stack(self):
        model_resource = self.model_resource
        start_layer = model_resource[2][0]
        layer_stack = [start_layer]
        temp_stack = [start_layer]
        while len(layer_stack) < len(model_resource[0]):
            start_layer = temp_stack.pop(0)
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    node_check = True
                    for pa in model_resource[1][ch]["parent"]:
                        if pa not in layer_stack:
                            node_check = False
                            break
                    if node_check:
                        if ch not in layer_stack:
                            layer_stack.append(ch)
                    temp_stack.append(ch)
        self.layer_stack = layer_stack

    def create_every_model_output(self, model):
        class ModelWithEveryOutputs(nn.Module):
            def __init__(self, base_model):
                super(ModelWithEveryOutputs, self).__init__()
                self.base_model = base_model
            def forward(self, x):
                outputs = []
                for layer_name, layer in self.base_model._modules.items():
                    if isinstance(x, tuple):
                        if isinstance(layer, nn.LSTM):
                            # Assuming you want to take the last LSTM output
                            x, _ = layer(x[0])  # Pass the first element of the tuple (assumes one LSTM layer)
                        else:
                            x = layer(x[0])  # Pass the first element of the tuple
                    else:
                        x = layer(x)
                    outputs.append((layer_name, x))
                return outputs
        self.every_out_model = ModelWithEveryOutputs(model)

    def create_model_output(self, model):
        class ModelWithOutputs(nn.Module):
            def __init__(self, base_model):
                super(ModelWithOutputs, self).__init__()
                self.base_model = base_model

            def forward(self, x):
                outputs = []
                for layer_name, layer in self.base_model._modules.items():
                    if isinstance(layer, nn.LSTM):
                        lstm_output, _ = layer(x)
                        if lstm_output.dim() == 3:
                            x = lstm_output[:, -1, :]  # Take the output of the last time step
                        else:
                            x = lstm_output
                    else:
                        x = layer(x)
                    outputs.append((layer_name, x))
                return outputs

        # all_out_model = ModelWithOutputs(model)
        self.all_out_model = ModelWithOutputs(model)
        model.eval()
        model_resource = self.model_resource
        self.layers = [[], []]
        for l in model_resource[0]:
            self.layers[0].append(l)
            self.layers[1].append(model_resource[0][l])

    def predict_every(self, inputs):
        every_out = self.every_out_model(inputs)
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        every_temp_out = {}
        for i in range(len(every_out)):
            current_layer, current_layer_obj = every_out[i]
            try:
                next_layer, next_layer_obj = every_out[i + 1]
                current_layer_name = current_layer
                next_layer_name = next_layer
                next_layer_type = next_layer_name.lower()
                if any(af in next_layer_type for af in activation_functions):
                    if isinstance(next_layer_obj, tuple):
                        # Assuming you want the first tensor from the tuple
                        next_layer_tensor = next_layer_obj[0]
                    else:
                        next_layer_tensor = next_layer_obj
                    every_temp_out[
                        f"{current_layer_name}/{next_layer_name}"] = next_layer_tensor.detach().numpy().astype(
                        np.float32)
                    i += 1
                else:
                    if any(af in current_layer_name for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        every_temp_out[current_layer_name] = current_layer_tensor.detach().numpy().astype(np.float32)
            except:
                if any(af in next_layer_type for af in activation_functions):
                    pass
                else:
                    if any(af in current_layer for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        every_temp_out[current_layer] = current_layer_tensor.detach().cpu().numpy().astype(np.float32)
        return every_temp_out

    def predict(self, inputs):
        all_out = self.all_out_model(inputs)
        activation_functions = ['relu', 'sigmoid', 'tanh', 'softmax']
        temp_out = {}
        for i in range(len(all_out)):
            current_layer, current_layer_obj = all_out[i]
            try:
                next_layer, next_layer_obj = all_out[i + 1]
                current_layer_name = current_layer
                next_layer_name = next_layer
                next_layer_type = next_layer_name.lower()
                if any(af in next_layer_type for af in activation_functions):
                    if isinstance(next_layer_obj, tuple):
                        # Assuming you want the first tensor from the tuple
                        next_layer_tensor = next_layer_obj[0]
                    else:
                        next_layer_tensor = next_layer_obj
                    temp_out[
                        f"{current_layer_name}/{next_layer_name}"] = next_layer_tensor.detach().cpu().numpy().astype(
                        np.float32)
                    i += 1
                else:
                    if any(af in current_layer_name for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        temp_out[current_layer_name] = current_layer_tensor.detach().numpy().astype(np.float32)
            except:
                if any(af in next_layer_type for af in activation_functions):
                    pass
                else:
                    if any(af in current_layer for af in activation_functions) is False:
                        if isinstance(current_layer_obj, tuple):
                            # Assuming you want the first tensor from the tuple
                            current_layer_tensor = current_layer_obj[0]
                        else:
                            current_layer_tensor = current_layer_obj
                        temp_out[current_layer] = current_layer_tensor.detach().cpu().numpy().astype(np.float32)
        return temp_out

    def eval(
            self,
            all_out,
            mode="default",
            start_wt=[],
            multiplier=100.0,
            scaler=0,
            max_unit=0,
            predicted_token=None,
            thresholding=0.5,
            task="binary-classification",
            use_refactored_handlers: bool = False
    ):
        if mode == "default":
            output = self.proportional_eval(
                all_out=all_out,
                start_wt=start_wt,
                multiplier=multiplier,
                scaler=scaler,
                max_unit=max_unit,
                predicted_token=predicted_token,
                thresholding=thresholding,
                task=task,
                use_refactored_handlers=use_refactored_handlers
            )
            return output
        elif mode == "contrast":
            temp_output = self.contrast_eval(
                all_out=all_out, 
                multiplier=multiplier,
                scaler=0,
                thresholding=0.5,
                task="binary-classification",
            )
            output = {}
            for k in temp_output[0].keys():
                output[k] = {}
                output[k]["Positive"] = temp_output[0][k]
                output[k]["Negative"] = temp_output[1][k]
            return output

    def proportional_eval(
            self, all_out, start_wt=[], multiplier=100.0, 
            scaler=0, max_unit=0, predicted_token=None,
            thresholding=0.5, task="binary-classification",
            use_refactored_handlers: bool = False
    ):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        out_layer = model_resource[2][0]
        all_wt = {}

        ActiveUP = UP_REFACTORED if use_refactored_handlers else UP_ORIGINAL

        if len(start_wt) == 0:
            if self.model_type == 'encoder':
                _start_wt_np = UP_ORIGINAL.calculate_start_wt(all_out[out_layer].detach().numpy(), scaler=scaler)
                all_wt[out_layer] = _start_wt_np * multiplier
            elif self.model_type == 'encoder_decoder':
                _start_wt_np = UP_ORIGINAL.calculate_enc_dec_start_wt(all_out[out_layer][0].detach().numpy(), predicted_token)
                all_wt[out_layer] = _start_wt_np * multiplier
            else:
                _start_wt_np = UP_ORIGINAL.calculate_start_wt(all_out[out_layer], scaler, thresholding, task=task)
                all_wt[out_layer] = _start_wt_np * multiplier
            
                layer_stack = self.layer_stack
            if self.model_type in ['encoder', 'encoder_decoder']:
                all_wts = self.model_weights
            else:
                all_wts = None
                
        for start_layer in tqdm(layer_stack):
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt:
                        if model_resource[1][start_layer]["class"] == 'LSTM':
                            if isinstance(all_out[ch], tuple):
                                all_wt[ch] = np.zeros_like(all_out[ch][0])
                        else:
                                all_wt[ch] = np.zeros_like(all_out[ch])
                    else:
                        if isinstance(all_out[ch], tuple):
                            all_wt[ch] = np.zeros_like(all_out[ch][0])
                        else:
                            all_wt[ch] = np.zeros_like(all_out[ch])

                module_obj = model_resource[0][start_layer]
                layer_name_for_activation = model_resource[1][start_layer]["name"]
                activation_conf = activation_dict.get(layer_name_for_activation, activation_master.get("None"))

                current_relevance_np = all_wt[start_layer] 
                
                input_activation_np = None
                if child_nodes:
                    input_activation_np = all_out[child_nodes[0]]

                if model_resource[1][start_layer]["class"] == "Linear":
                    weights_t = module_obj.weight
                    bias_t = module_obj.bias
                    
                    if use_refactored_handlers:
                        temp_wt = ActiveUP.calculate_fc_input_relevance_pytorch_cuda(
                            current_relevance_np,
                            input_activation_np,
                            weights_t,
                            bias_t,
                            activation_conf
                        )
                    else:
                         temp_wt = ActiveUP.calculate_wt_fc(
                            current_relevance_np,
                            input_activation_np,
                            weights_t,
                            bias_t,
                            activation_conf
                        )
                    if use_refactored_handlers and isinstance(temp_wt, torch.Tensor):
                        temp_wt = temp_wt.cpu().detach().numpy()
                    all_wt[child_nodes[0]] += temp_wt

                elif model_resource[1][start_layer]["class"] == "Conv2d":
                    weights_t = module_obj.weight
                    bias_t = module_obj.bias
                    padding_val = module_obj.padding
                    stride_val = module_obj.stride
                    
                    if use_refactored_handlers:
                        # Ensure stride_val and padding_val are in correct format for CUDA function
                        if isinstance(stride_val, int):
                            stride_val_tuple = (stride_val, stride_val)
                        else:
                            stride_val_tuple = stride_val # Should be (sH, sW)
                        
                        # padding_val can be int, tuple, or string 'same'/'valid'
                        # calculate_padding_pytorch_cuda handles these types.
                        padding_config_for_cuda = padding_val

                        # Prepare inputs as NumPy arrays first for consistent transposition logic
                        np_input_act_hwc = np.transpose(input_activation_np.squeeze(0), (1, 2, 0)) # InH, InW, InC
                        np_kernel_khw_inc_outc = np.transpose(weights_t.cpu().detach().numpy(), (2, 3, 1, 0)) # kH, kW, InC, OutC
                        np_relevance_ohw_outc = np.transpose(current_relevance_np.squeeze(0), (1, 2, 0)) # OutH, OutW, OutC
                        np_bias_outc = bias_t.cpu().detach().numpy() # OutC

                        # Convert to PyTorch tensors for the CUDA function
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        dtype = torch.float32

                        torch_input_ihwc = torch.as_tensor(np_input_act_hwc, device=device, dtype=dtype)
                        torch_kernel_khw_inc_outc = torch.as_tensor(np_kernel_khw_inc_outc, device=device, dtype=dtype)
                        torch_bias_outc = torch.as_tensor(np_bias_outc, device=device, dtype=dtype)
                        torch_relevance_ohw_outc = torch.as_tensor(np_relevance_ohw_outc, device=device, dtype=dtype)

                        temp_wt_hwc_tensor = ActiveUP.calculate_wt_conv_pytorch_cuda(
                            input_tensor_ihwc=torch_input_ihwc,
                            kernel_filters_tensor_khw_inc_outc=torch_kernel_khw_inc_outc,
                            kernel_bias_tensor_outc=torch_bias_outc,
                            output_channel_gain_map_tensor_ohw_outc=torch_relevance_ohw_outc,
                            strides_hw=stride_val_tuple,
                            padding_mode_or_values=padding_config_for_cuda,
                            activation_config=activation_conf
                        )
                        # Convert result back to NumPy for subsequent operations
                        temp_wt_hwc = temp_wt_hwc_tensor.cpu().detach().numpy()
                        temp_wt = np.transpose(temp_wt_hwc, (2,0,1))[np.newaxis, ...]

                    else:
                        temp_wt = ActiveUP.calculate_wt_conv(
                            current_relevance_np,
                            input_activation_np,
                            weights_t,
                            bias_t,
                            padding_val,
                            stride_val,
                            activation_conf
                        )
                    if use_refactored_handlers and isinstance(temp_wt, torch.Tensor):
                        pass
                    elif not use_refactored_handlers and isinstance(temp_wt, torch.Tensor):
                        temp_wt = temp_wt.cpu().detach().numpy()
                    all_wt[child_nodes[0]] += temp_wt

                elif model_resource[1][start_layer]["class"] == "MaxPool2d":
                    padding_val = module_obj.padding
                    stride_val = module_obj.stride
                    kernel_size_val = module_obj.kernel_size
                    
                    if isinstance(kernel_size_val, int):
                        kernel_size_val = (kernel_size_val, kernel_size_val)

                    if use_refactored_handlers:
                        # Ensure stride_val is a tuple for the CUDA function
                        if isinstance(stride_val, int):
                            stride_val_tuple = (stride_val, stride_val)
                        else:
                            stride_val_tuple = stride_val

                        # Prepare inputs as NumPy arrays first for consistent transposition logic
                        np_input_act_hwc = np.transpose(input_activation_np.squeeze(0), (1, 2, 0))
                        np_relevance_ohw_outc = np.transpose(current_relevance_np.squeeze(0), (1, 2, 0))

                        # Convert to PyTorch tensors for the CUDA function
                        # Assuming float32 and current CUDA device
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        dtype = torch.float32

                        torch_input_act_hwc = torch.as_tensor(np_input_act_hwc, device=device, dtype=dtype)
                        torch_relevance_ohw_outc = torch.as_tensor(np_relevance_ohw_outc, device=device, dtype=dtype)

                        temp_wt_hwc_tensor = ActiveUP.calculate_wt_maxpool_pytorch_cuda(
                            torch_input_act_hwc,
                            torch_relevance_ohw_outc,
                            kernel_size_val,
                            stride_val_tuple, # Pass the ensured tuple
                            padding_val, # mode string or tuple/list
                        )
                        # Convert result back to NumPy for subsequent operations
                        temp_wt_hwc = temp_wt_hwc_tensor.cpu().detach().numpy()
                        temp_wt = np.transpose(temp_wt_hwc, (2,0,1))[np.newaxis, ...]
                    else:
                        temp_wt = ActiveUP.calculate_wt_maxpool(
                            current_relevance_np,
                            input_activation_np,
                            kernel_size_val,
                            padding_val,
                            stride_val
                        )
                    if use_refactored_handlers and isinstance(temp_wt, torch.Tensor):
                        pass
                    elif not use_refactored_handlers and isinstance(temp_wt, torch.Tensor):
                        temp_wt = temp_wt.cpu().detach().numpy()
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] == "Flatten":
                    if use_refactored_handlers:
                        # Ensure inputs are tensors for the refactored CUDA version
                        current_relevance_t = torch.as_tensor(current_relevance_np, device='cuda' if torch.cuda.is_available() else 'cpu')
                        input_activation_t = torch.as_tensor(input_activation_np, device='cuda' if torch.cuda.is_available() else 'cpu')
                        temp_wt = ActiveUP.reshape_tensor_pytorch_cuda(
                            current_relevance_t, 
                            input_activation_t
                        )
                    else:
                        temp_wt = ActiveUP.calculate_wt_rshp(
                            current_relevance_np,
                            input_activation_np
                        )
                    if use_refactored_handlers and isinstance(temp_wt, torch.Tensor):
                        temp_wt = temp_wt.cpu().detach().numpy()
                    all_wt[child_nodes[0]] += temp_wt
                    
                elif model_resource[1][start_layer]["class"] in ["ReLU", "Identity", "Softmax"]:
                    all_wt[child_nodes[0]] += current_relevance_np
                
                else:
                    temp_wt = current_relevance_np
                    all_wt[child_nodes[0]] += temp_wt
        
        if max_unit > 0 and scaler == 0:
            temp_dict = {}
            for k_wt, v_wt in all_wt.items():
                temp_dict[k_wt] = UP_ORIGINAL.weight_normalize(v_wt, max_val=max_unit)
            all_wt = temp_dict
        elif scaler > 0:
            temp_dict = {}
            for k_wt, v_wt in all_wt.items():
                temp_dict[k_wt] = UP_ORIGINAL.weight_scaler(v_wt, scaler=scaler)
            all_wt = temp_dict

        return all_wt

    def contrast_eval(self, all_out, multiplier=100.0,
                            scaler=None,thresholding=0.5,
                            task="binary-classification"):
        model_resource = self.model_resource
        activation_dict = self.activation_dict
        inputcheck = False
        out_layer = model_resource[2][0]
        all_wt_pos = {}
        all_wt_neg = {}
        start_wt_pos, start_wt_neg = UC.calculate_start_wt(all_out[out_layer],scaler,thresholding,task)
        all_wt_pos[out_layer] = start_wt_pos * multiplier
        all_wt_neg[out_layer] = start_wt_neg * multiplier
        layer_stack = [out_layer]

        while len(layer_stack) > 0:
            start_layer = layer_stack.pop(0)
            if model_resource[1][start_layer]["child"]:
                child_nodes = model_resource[1][start_layer]["child"]
                for ch in child_nodes:
                    if ch not in all_wt_pos:
                        all_wt_pos[ch] = np.zeros_like(all_out[ch][0])
                        all_wt_neg[ch] = np.zeros_like(all_out[ch][0])
                if model_resource[1][start_layer]["class"] == "Linear":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_fc(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif model_resource[1][start_layer]["class"] == "Conv2d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_conv(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1,
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "ConvTranspose2d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv2d_transpose(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        w1,
                        b1,
                        pad1, 
                        strides1,
                        activation_dict[model_resource[1][start_layer]["name"]],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == 'Conv1d':
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding[0]
                    strides1 = l1.stride[0]
                    dilation1 = l1.dilation
                    groups1 = l1.groups
                    if not isinstance(b1, np.ndarray):
                        b1 = b1.numpy()
                    if not isinstance(w1, np.ndarray):
                        w1 = w1.numpy()  # Convert PyTorch tensor to NumPy array

                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv_1d(all_wt_pos[start_layer],
                                                                all_wt_neg[start_layer],
                                                                all_out[child_nodes[0]][0],
                                                                w1,b1, pad1, strides1,dilation1,groups1,
                                                                activation_dict[model_resource[1][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "ConvTranspose1d":
                    l1 = model_resource[0][start_layer]
                    w1 = l1.state_dict()['weight']
                    b1 = l1.state_dict()['bias']
                    pad1 = l1.padding[0]
                    strides1 = l1.stride[0]
                    temp_wt_pos,temp_wt_neg = UC.calculate_wt_conv1d_transpose(all_wt_pos[start_layer],
                                                                            all_wt_neg[start_layer],
                                                                            all_out[child_nodes[0]][0],
                                                                            w1,b1, pad1, strides1,
                                                                            activation_dict[model_resource[1][start_layer]['name']])
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "Reshape":
                    temp_wt_pos = UC.calculate_wt_rshp(
                        all_wt_pos[start_layer], all_out[child_nodes[0]][0]
                    )
                    temp_wt_neg = UC.calculate_wt_rshp(
                        all_wt_neg[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                elif (
                        model_resource[1][start_layer]["class"] == "AdaptiveAvgPool2d"
                ):
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_gavgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "Flatten":
                    temp_wt = UC.calculate_wt_rshp(
                        all_wt_pos[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt
                    temp_wt = UC.calculate_wt_rshp(
                        all_wt_neg[start_layer], all_out[child_nodes[0]][0]
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt
                elif (
                        model_resource[1][start_layer]["class"] == "AdaptiveAvgPool2d"
                ):
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_gavgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "MaxPool2d":
                    l1 = model_resource[0][start_layer]
                    temp_wt = UC.calculate_wt_maxpool(
                        all_wt_pos[start_layer],
                        all_out[child_nodes[0]][0],
                        (l1.kernel_size, l1.kernel_size),
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt.T
                    temp_wt = UC.calculate_wt_maxpool(
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        (l1.kernel_size, l1.kernel_size),
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "MaxPool1d":
                    l1 = model_resource[0][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt = UC.calculate_wt_maxpool_1d(
                        all_wt_pos[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.kernel_size, pad1, strides1
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt.T
                    temp_wt = UC.calculate_wt_maxpool_1d(
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.kernel_size, pad1, strides1
                    )
                    all_wt_neg[child_nodes[0]] += temp_wt.T
                elif model_resource[1][start_layer]["class"] == "AvgPool2d":
                    l1 = model_resource[0][start_layer]
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_avgpool(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        (l1.kernel_size, l1.kernel_size),
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "AvgPool1d":
                    l1 = model_resource[0][start_layer]
                    pad1 = l1.padding
                    strides1 = l1.stride
                    temp_wt_pos, temp_wt_neg = UC.calculate_wt_avgpool_1d(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        all_out[child_nodes[0]][0],
                        l1.kernel_size, pad1, strides1
                    )
                    all_wt_pos[child_nodes[0]] += temp_wt_pos.T
                    all_wt_neg[child_nodes[0]] += temp_wt_neg.T
                elif model_resource[1][start_layer]["class"] == "Concatenate":
                    temp_wt = UC.calculate_wt_concat(
                        all_wt_pos[start_layer],
                        [all_out[ch] for ch in child_nodes],
                        model_resource[0][start_layer].axis,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt_pos[ch] += temp_wt[ind]
                    temp_wt = UC.calculate_wt_concat(
                        all_wt_neg[start_layer],
                        [all_out[ch] for ch in child_nodes],
                        model_resource[0][start_layer].axis,
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt_neg[ch] += temp_wt[ind]
                elif model_resource[1][start_layer]["class"] == "Add":
                    temp_wt = UC.calculate_wt_add(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        [all_out[ch] for ch in child_nodes],
                    )
                    for ind, ch in enumerate(child_nodes):
                        all_wt_pos[ch] += temp_wt[ind][0]
                        all_wt_neg[ch] += temp_wt[ind][1]
                elif model_resource[1][start_layer]["class"] == "LSTM":
                    l1 = model_resource[0][start_layer]
                    return_sequence = l1.return_sequences
                    units = l1.units
                    num_of_cells = l1.input_shape[1]
                    lstm_obj_f = UC.LSTM_forward(
                        num_of_cells, units, l1.weights, return_sequence, False
                    )
                    lstm_obj_b = UC.LSTM_backtrace(
                        num_of_cells,
                        units,
                        [i.numpy() for i in l1.weights],
                        return_sequence,
                        False,
                    )
                    temp_out_f = lstm_obj_f.calculate_lstm_wt(
                        all_out[child_nodes[0]][0]
                    )
                    temp_wt_pos, temp_wt_neg = lstm_obj_b.calculate_lstm_wt(
                        all_wt_pos[start_layer],
                        all_wt_neg[start_layer],
                        lstm_obj_f.compute_log,
                    )
                    all_wt_pos[child_nodes[0]] = temp_wt_pos
                    all_wt_neg[child_nodes[0]] = temp_wt_neg
                elif model_resource[1][start_layer]["class"] == "Embedding":
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]

                    temp_wt_pos = np.mean(temp_wt_pos,axis=1)
                    temp_wt_neg = np.mean(temp_wt_neg,axis=1)

                    all_wt_pos[child_nodes[0]] = all_wt_pos[child_nodes[0]] + temp_wt_pos
                    all_wt_neg[child_nodes[0]] = all_wt_neg[child_nodes[0]] + temp_wt_neg
                else:
                    temp_wt_pos = all_wt_pos[start_layer]
                    temp_wt_neg = all_wt_neg[start_layer]
                    all_wt_pos[child_nodes[0]] += temp_wt_pos
                    all_wt_neg[child_nodes[0]] += temp_wt_neg
                for ch in child_nodes:
                    if not (ch in layer_stack):
                        layer_stack.append(ch)
        return all_wt_pos, all_wt_neg
