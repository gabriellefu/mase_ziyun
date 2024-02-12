import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

# figure out the correct path
machop_path = Path("/workspaces/mase/machop")
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-three-layer"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

from torch import nn
from chop.passes.graph.utils import get_parent_name


class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)

# generate the mase graph and initialize node metadata
model = JSC_Three_Linear_Layers()
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)

    


def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
        main_config = pass_args
        print(main_config.pop('default', None))
        i = 0
        last_output=0
        for node in graph.fx_graph.nodes:
            i += 1
            # if node name is not matched, it won't be tracked
            config = main_config.get(node.name, None)
            if config!=None:
                config=config["config"]
                name = config.get("name", None)
            else:
                name = None
            # print(node.meta["mase"].parameters["common"].keys())
            if name is not None:
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
            elif  node.target=="relu":
                        new_module=nn.ReLU()
                        parent_name, name = get_parent_name(node.target)
                        setattr(graph.modules[parent_name], name, new_module)

        return graph, {}




pass_config = {
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
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

# this performs the architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})

##building search space
import copy
import numpy as np
#the input size simplifier of the 2nd layer output is the same as the 4th layer input
#and the simplifier of the 4th layer output is the same of the 6th layer input, thus we only need to choose the value of two simplifier
mulitiplier=[2,3,4,5,6]
search_spaces = []
for config in mulitiplier:
        pass_ori=pass_config
        pass_ori['seq_blocks_2']['config']['channel_multiplier'] = config
        pass_ori['seq_blocks_4']['config']['channel_multiplier']=config
        pass_ori['seq_blocks_6']['config']['channel_multiplier'] = config
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_config))


import torch
from torchmetrics.classification import MulticlassAccuracy
from chop.passes.graph import report_graph_analysis_pass
model=JSC_Three_Linear_Layers()
from chop.passes.graph import report_graph_analysis_pass
mg_ori = MaseGraph(model=model)
mg_ori, _ = init_metadata_analysis_pass(mg_ori, None)
mg_ori, _ = add_common_metadata_analysis_pass(mg_ori, {"dummy_in": dummy_in})
mg_ori, _ = add_software_metadata_analysis_pass(mg_ori, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search
recorded_accs = []
for i, config in enumerate(search_spaces):
    
    mg, _ = redefine_linear_transform_pass(
    graph=mg_ori, pass_args=config)
    j = 0
    model=mg.model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    max_epoch=1
    for epoch in range(max_epoch):
        print("epoch",epoch)
        for data, labels in data_module.train_dataloader():
            outputs =model(data)
            loss = criterion(outputs,labels)
            #print(loss)
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()
        
    for input in data_module.train_dataloader():
        xs, ys = input
        preds = model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    print(f"The size is network is {config['seq_blocks_4']['config']['channel_multiplier']} times of its original network.")
    print(" Accuracy of this network is ",acc_avg.item())
    print()
    recorded_accs.append(acc_avg)

