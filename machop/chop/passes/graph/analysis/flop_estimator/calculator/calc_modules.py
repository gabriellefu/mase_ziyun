import torch
import numpy as np


def calculate_modules(module, in_data, out_data):
    # Collect computation statistics.
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        # One computation per input pixel - window size is chosen adaptively
        # and windows never overlap (?).
        assert len(in_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        computations = input_size
        backward_computations = input_size
        if (np.dtype(input_size) == "float16") or (np.dtype(input_size) == "int16"):
            multiple=16
        elif  np.dtype(input_size) == "int8":
            multiple=8
        elif (np.dtype(input_size) == "float32") or (np.dtype(input_size) == "int32"):
            multiple=32
        elif (np.dtype(input_size) == "float64") or (np.dtype(input_size) == "int64"):
            multiple=64
        
        
        return {
            "bit_computations":computations*multiple,
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Embedding):
        total_parameters = module.embedding_dim * in_data[0].numel()
        return {
            "bit_computations":0,
            "total_parameters": total_parameters,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": 0,
            "output_buffer_size": 0,
        }
    elif isinstance(module, torch.nn.AvgPool2d) or isinstance(
        module, torch.nn.MaxPool2d
    ):
        # Each output pixel requires computations on a 2D window of input.
        if type(module.kernel_size) == int:
            # Kernel size here can be either a single int for square kernel
            # or a tuple (see
            # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d )
            window_size = module.kernel_size**2
        else:
            window_size = module.kernel_size[0] * module.kernel_size[1]

        # Not sure which output tensor to use if there are multiple of them.
        assert len(out_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        computations = output_size * window_size
        backward_computations = input_size * window_size
        if (np.dtype(output_size) == "float16") or (np.dtype(output_size) == "int16"):
            multiple=16
        elif  np.dtype(output_size) == "int8":
            multiple=8
        elif (np.dtype(output_size) == "float32") or (np.dtype(output_size) == "int32"):
            multiple=32
        elif (np.dtype(output_size) == "float64") or (np.dtype(output_size) == "int64"):
            multiple=64
        return {
            "bit_computations":computations*multiple,
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Conv2d):
        # Each output pixel requires computations on a 3D window of input.
        # Not sure which input tensor to use if there are multiple of them.
        assert len(in_data) == 1
        _, channels, _, _ = in_data.size()
        window_size = module.kernel_size[0] * module.kernel_size[1] * channels

        # Not sure which output tensor to use if there are multiple of them.
        assert len(out_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()

        computations = output_size * window_size
        backward_computations = input_size * window_size * 2
        if (np.dtype(output_size) == "float16") or (np.dtype(output_size) == "int16"):
            multiple=16
        elif  np.dtype(output_size) == "int8":
            multiple=8
        elif (np.dtype(output_size) == "float32") or (np.dtype(output_size) == "int32"):
            multiple=32
        elif (np.dtype(output_size) == "float64") or (np.dtype(output_size) == "int64"):
            multiple=64
        return {
            "bit_computations":computations*multiple,
            "total_parameters": module.weight.numel(),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Dropout2d) or isinstance(
        module, torch.nn.modules.dropout.Dropout
    ):
        return {
            "bit_computations":0,
            "total_parameters": 0,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.Linear):
        # One computation per weight, for each batch element.

        # Not sure which input tensor to use if there are multiple of them.
        # TODO: check if this is correct
        # TODO: also consider bias?
        assert len(in_data) == 1
        batch = in_data[0].numel() / in_data[0].shape[-1]

        computations = module.weight.numel() * batch
        backward_computations = module.weight.numel() * batch * 2
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        if module.weight.dtype == torch.float16 or module.weight.dtype == torch.int16:
            multiple = 16
        elif module.weight.dtype == torch.int8:
            multiple = 8
        elif module.weight.dtype == torch.float32 or module.weight.dtype == torch.int32:
            multiple = 32
        elif module.weight.dtype == torch.float64 or module.weight.dtype == torch.int64:
            multiple = 64
        return {
            "bit_computations":computations*multiple,
            "total_parameters": module.weight.numel(),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(
        module, torch.nn.modules.activation.ReLU6
    ):

        if in_data[0].dtype == torch.float16 or in_data[0].dtype == torch.int16:
            multiple = 16
        elif in_data[0].dtype == torch.int8:
            multiple = 8
        elif in_data[0].dtype == torch.float32 or in_data[0].dtype == torch.int32:
            multiple = 32
        elif in_data[0].dtype == torch.float64 or in_data[0].dtype == torch.int64:
            multiple = 64
        # ReLU does a single negation check
        return {
            "bit_computations":in_data[0].numel()*multiple,
            "total_parameters": 0,
            "computations": in_data[0].numel(),
            "backward_computations": in_data[0].numel(),
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.LayerNorm):
        if (np.dtype( in_data[0]) == "float16") or (np.dtype( in_data[0]) == "int16"):
            multiple=16
        elif  np.dtype( in_data[0]) == "int8":
            multiple=8
        elif (np.dtype( in_data[0]) == "float32") or (np.dtype( in_data[0]) == "int32"):
            multiple=32
        elif (np.dtype( in_data[0]) == "float64") or (np.dtype( in_data[0]) == "int64"):
            multiple=64
        return {
            "bit_computations":in_data[0].numel()*multiple* 5,
            "total_parameters": 0,
            "computations": in_data[0].numel() * 5,
            "backward_computations": in_data[0].numel() * 5,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d) or isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        # Accesses to E[x] and Var[x] (all channel size)
        total_parameters = 2 * module.num_features
        # (x-running_mean)/running variance
        # multiply by gamma and beta addition
        computations = 4 * in_data[0].numel()
        backward_computations = 4 * in_data[0].numel()
        if isinstance(in_data, torch.Tensor):
            if in_data.dtype == torch.float16 or in_data.dtype == torch.int16:
                multiple = 16
            elif in_data.dtype == torch.int8:
                multiple = 8
            elif in_data.dtype == torch.float32 or in_data.dtype == torch.int32:
                multiple = 32
            elif in_data.dtype == torch.float64 or in_data.dtype == torch.int64:
                multiple = 64
        elif isinstance(in_data, list):

            multiple = 1 
        elif isinstance(in_data, np.ndarray):
            if in_data.dtype == np.float16 or in_data.dtype == np.int16:
                multiple = 16
            elif  np.dtype( in_data) == "int8":
                multiple=8
            elif (np.dtype( in_data) == "float32") or (np.dtype( in_data) == "int32"):
                multiple=32
            elif (np.dtype( in_data) == "float64") or (np.dtype( in_data) == "int64"):
                multiple=64
        # if (np.dtype( in_data[0]) == "float16") or (np.dtype( in_data[0]) == "int16"):
        #     multiple=16
        # elif  np.dtype( in_data[0]) == "int8":
        #     multiple=8
        # elif (np.dtype( in_data[0]) == "float32") or (np.dtype( in_data[0]) == "int32"):
        #     multiple=32
        # elif (np.dtype( in_data[0]) == "float64") or (np.dtype( in_data[0]) == "int64"):
        #     multiple=64
        return {
            "bit_computations":computations*multiple,
            "total_parameters": total_parameters,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }
    else:
        print("Unsupported module type for analysis:", type(module))




