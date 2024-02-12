1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.
>* In this task, we need to modify the redefine_linear_transform_pass() function to modify "relu" node when the size of the output of the former node changes.
>* We also need to set a correct pass to modify the network and expanded it's size.
>* The code for redefine_linear_transform_pass() is shown as follows. We add a condition that when node.meta["mase"].parameters["common"]["mase_op"] is "relu", we need to create a new_module and connect it with the now graph.
```python
def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        # print(node.meta["mase"].parameters["common"].keys())
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
        elif   node.meta["mase"].parameters["common"]["mase_op"] == "relu":
            new_module=nn.ReLU()
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)

    return graph, {}
```
>* The network struture before and after modifying are shown as follows. As is shown in the images, the size of each layer has become the size of its original size.
![alt text](lab4_1out1.png)
![alt text](lab4_1out2.png)

2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

>* Firstly, we need to define a search space. We test the network with each layer have the size of its original size's of 2,3,4,5 and 6 times.
```python
mulitiplier=[2,3,4,5,6]
search_spaces = []
for config in mulitiplier:
        pass_config['seq_blocks_2']['config']['channel_multiplier'] = config
        pass_config['seq_blocks_4']['config']['channel_multiplier']=config
        pass_config['seq_blocks_6']['config']['channel_multiplier'] = config
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_config))
```
>* Then we use accurecy as the quality metric in this task and record the average accuracy in each network.
>* The result shows as follows, and it shows that the best performance occurs when the network is 2 times of original network.
![alt text](lab4_2.png)


3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following:
```python
# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 32),  # output scaled by 2
            nn.ReLU(32),  # scaled by 2
            nn.Linear(32, 64),  # input scaled by 2 but output scaled by 4
            nn.ReLU(64),  # scaled by 4
            nn.Linear(64, 5),  # scaled by 4
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

>* In this task, we notice that the input size of a node must aligns with the output size with the output in the former layer. In this case, when we want to change both the input_feature size and the output_feature size, we only need to choose one value for its output size and set its input size the same as the output size of the layer before.
>* For the same reason, when we only need to change the input size, actually we don't need to choose a specific value of this input size. 
>* The code of the new function is shown as follows. We follow the rules above and save the output size of the last node and used it as the input size of the node when it needed to be modified.
```python
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
            elif   node.meta["mase"].parameters["common"]["mase_op"] == "relu":
                new_module=nn.ReLU()
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)

        return graph, {}
```
>* Then we set the config of the new pass as follows.
```python
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
        "name": "both_different",
        "channel_multiplier": 4,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 4,
        }
    },
}
```
>* After apply the function to the graph and report the original and modified graph, the result is shown as follows which indicating that the network has been changed as required.
![alt text](lab4_3result.png)

4. Integrate the search to the chop flow, so we can run it from the command line.
>* We firstly made a new toml file for the configuration of the search in /workspaces/mase/machop/configs/examples/lab4_ziyun.toml.
>* Then we bulid a search space GraphSearchSpaceMulitiplier(SearchSpaceBase) and transmit the configuration in each search to redefine_linear_transform_pass() in /workspaces/mase/machop/chop/actions/search/search_space/multiplier/graph_ziyun.py.
>* We also modify /workspaces/mase/machop/chop/actions/search/strategies/runners/software/train.pyand return the validation loss and accuracy in each structure.
>* The seach is integrate to the chop flow and we can used the cmd "./ch search --config /workspaces/mase/machop/configs/examples/lab4_ziyun.toml --load /workspaces/mase/mase_output/jsc-three-layer_classification_jsc_2024-02-11/software/training_ckpts/best.ckpt" to run it.
>* The result of search is shown as follows.

|    | number | software_metrics                   | hardware_metrics                                | scaled_metrics                                |
|----|--------|-------------------------------------|-------------------------------------------------|-----------------------------------------------|
| 0  | 1      | {'loss': 1.607, 'accuracy': 0.233} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.233, 'average_bitwidth': 6.4} |


>* The config structure of the best network is shown as follows.
```python
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 48),  # output scaled by 3
            nn.ReLU(48),  # scaled by 3
            nn.Linear(48, 32),  # input scaled by 3 but output scaled by 2
            nn.ReLU(32),  # scaled by 2
            nn.Linear(32, 5),  # scaled by 2
            nn.ReLU(5),
        )
```