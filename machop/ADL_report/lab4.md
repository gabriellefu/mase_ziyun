Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.
We need to modify the pass with the following code.
![alt text](image-4.png)
Then, we need to adding the condition to tranform "relu" layer to adapt to the output of last layer.
![alt text](image-3.png)

output:
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    return seq_blocks_5
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 6, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=32, bias=True), ReLU(inplace=True), Linear(in_features=32, out_features=32, bias=True), ReLU(inplace=True), Linear(in_features=32, out_features=5, bias=True), ReLU(inplace=True)]




task3:Can you then design a search so that it can reach a network that can have this kind of structure?
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    return seq_blocks_5
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 6, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=32, bias=True), ReLU(), Linear(in_features=32, out_features=64, bias=True), ReLU(), Linear(in_features=64, out_features=5, bias=True), ReLU()]