from .utils.prop import np_swish

activation_master = {
    "None": {
        "name": "None",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0},
        "range": {"l": None, "u": None},
        "type": "mono",
        "func": None,
    },
    "linear": {
        "name": "linear",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0},
        "range": {"l": None, "u": None},
        "type": "mono",
        "func": None,
    },
    "tanh": {
        "name": "tanh",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0},
        "range": {"l": -2, "u": 2},
        "type": "mono",
        "func": None,
    },
    "sigmoid": {
        "name": "sigmoid",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0},
        "range": {"l": -4, "u": 4},
        "type": "mono",
        "func": None,
    },
    "relu": {
        "name": "relu",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0},
        "range": {"l": 0, "u": None},
        "type": "mono",
        "func": None,
    },
    "swish": {
        "name": "swish",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0, "beta": 0.75},
        "range": {"l": -6, "u": None},
        "type": "non_mono",
        "func": np_swish,
    },
    "softmax": {
        "name": "softmax",
        "params": {"mul_val": 1.0, "const_val": 0.0, "add_val": 0.0},
        "range": {"l": -1, "u": 2},
        "type": "mono",
        "func": None,
    },
}
