# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.multiplier import (
    MULTIPLIER_OP,
    mulitplier_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict


from torch import nn
from chop.passes.graph.utils import get_parent_name



DEFAULT_MULITIPLE_CONFIG = {
    "config":{
        "by": "name",
        "default": {"config": {"name": None}},
        "seq_blocks_2": {
            "config": {
                "name": "output_only",
                # weight
                "channel_multiplier": 2,
                }
            },
        "seq_blocks_4": {
            "config": {
                "name": "both_different",
                "channel_multiplier": [2 ,4],
                }
            },
        "seq_blocks_6": {
            "config": {
                "name": "input_only",
                "channel_multiplier": 4,
                }
            },

    }

}

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)


class GraphSearchSpaceMulitiplier(SearchSpaceBase):

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_MULITIPLE_CONFIG

        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph

        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval() 
        else:
            self.model.train()

        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg, _ = init_metadata_analysis_pass(mg, None)
            mg, _ = add_common_metadata_analysis_pass(
                mg, {"dummy_in": self.dummy_input}
            )
            #self.mg = mg
        if sampled_config is not None:
            mg, _ = redefine_linear_transform_pass(mg, pass_args= sampled_config)
        mg.model.to(self.accelerator)
        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """
        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            #print("name=",node.name)
            node_info[node.name] = {
                "name": node.name,
                "mase_op":get_mase_op(node)

            }

        # Build the search space
        choices = {}
        seed = self.config["seed"]
        print(seed)

        match self.config["setup"]["by"]:
            case "name":
                for n_name, n_info in node_info.items():
                    if n_info["mase_op"] in MULTIPLIER_OP:
                        if n_name in seed:
                            choices[n_name] = deepcopy(seed[n_name])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case _:
                raise ValueError(
                    f"Unknown quantization by: {self.config['setup']['by']}"
                )

        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]
        return config
    






def redefine_linear_transform_pass(graph, pass_args=None):
        main_config = pass_args
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError(f"default value must be provided.")
        i = 0
        last_output=0
        for node in graph.fx_graph.nodes:
            i += 1
            # if node name is not matched, it won't be tracked
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            # print(node.meta["mase"].parameters["common"].keys())
            linear_flag=0

            if name is not None:
                if (node.meta["mase"].parameters["common"]["mase_op"] == "linear" )and (linear_flag==0):
                    linear_flag=1
                ori_module = graph.modules[node.target]
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                elif name == "input_only":
                    in_features = last_output
                elif name == "both":
                    muliplier_output=config["channel_multiplier"]
                    in_features = last_output
                    out_features = out_features * muliplier_output
                last_output=out_features


                new_module = instantiate_linear(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
            elif   node.meta["mase"].parameters["common"]["mase_op"] == "relu":
                new_module=nn.ReLU()
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)

        return graph, {}
