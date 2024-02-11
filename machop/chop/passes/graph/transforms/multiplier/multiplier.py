from copy import copy, deepcopy
import logging
import torch
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from .modify import create_new_fn, create_new_module
from .quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param
from .summary import graph_iterator_compare_nodes, graph_iterator_node_histogram

logger = logging.getLogger(__name__)


MULTIPLIER_OP = (
    "linear"
)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


# def update_quant_meta_param(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass



def graph_iterator_multiplier_by_name(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in MULTIPLIER_OP:
            continue
        node_config = get_config(config, node.name)
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        output_layers_names = node_config.get("additional_layers_outputs", [])
        output_layers = [
            get_node_target_by_name(graph, name) for name in output_layers_names
        ]
        input_layers_names = node_config.get("additional_layers_inputs", [])
        input_layers = [
            get_node_target_by_name(graph, name) for name in input_layers_names
        ]
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                input_layers=input_layers,
                output_layers=output_layers,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
            logger.debug(f"Quantized module: {node.target} with config: {node_config}")
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(get_mase_type(node))
            )
    return graph



def quantize_transform_pass(graph, pass_args=None):
    """
    Apply quantization transformation to the given graph.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.


    - pass_args
        - by -> str : different quantization schemes choose from ["type", "name", "regx_name"]
    """

    by = pass_args.pop("by")
    match by:
        case "name":
            graph = graph_iterator_multiplier_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}



def mulitplier_transform_pass(graph, pass_args=None):
    """
    Apply quantization transformation to the given graph.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.


    - pass_args
        - by -> str : different quantization schemes choose from ["type", "name", "regx_name"]
    """

    by = pass_args.pop("by")
    match by:
        case "name":
            graph = graph_iterator_multiplier_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}

