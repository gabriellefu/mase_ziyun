"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn


class JSC_Toy(nn.Module):
    def __init__(self, info):
        super(JSC_Toy, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 8),  # linear              # 2
            nn.BatchNorm1d(8),  # output_quant       # 3
            nn.ReLU(8),  # 4
            # 2nd LogicNets Layer
            nn.Linear(8, 8),  # 5
            nn.BatchNorm1d(8),  # 6
            nn.ReLU(8),  # 7
            # 3rd LogicNets Layer
            nn.Linear(8, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)


class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self, info):
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
    


class JSC_ziyun_lab1(nn.Module):
    def __init__(self, info):
        super(JSC_ziyun_lab1, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 12),  # linear              # 2
            nn.BatchNorm1d(12),  # output_quant       # 3
            nn.ReLU(12),  # 4


            nn.Linear(12, 32),  # 5
            nn.BatchNorm1d(32),  # 6
            nn.ReLU(32),  # 7


            nn.Linear(32, 64),  # 5
            nn.BatchNorm1d(64),  # 6
            nn.ReLU(64),  # 7


            nn.Linear(64, 32),  # 5
            nn.BatchNorm1d(32),  # 6
            nn.ReLU(32),  # 7

            nn.Linear(32, 12),  # 5
            nn.BatchNorm1d(12),  # 6
            nn.ReLU(12),  # 7

            nn.Linear(12, 8),  # 5
            nn.BatchNorm1d(8),  # 6
            nn.ReLU(8),  # 7

            nn.Linear(8, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)


class JSC_Tiny(nn.Module):
    def __init__(self, info):
        super(JSC_Tiny, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 5),  # linear              # 2
            # nn.BatchNorm1d(5),  # output_quant       # 3
            nn.ReLU(5),  # 4
        )

    def forward(self, x):
        return self.seq_blocks(x)


class JSC_S(nn.Module):
    def __init__(self, info):
        super(JSC_S, self).__init__()
        self.config = info
        self.num_features = self.config["num_features"]
        self.num_classes = self.config["num_classes"]
        hidden_layers = [64, 32, 32, 32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.ReLU()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, bn, out_act]
            else:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x


# Getters ------------------------------------------------------------------------------
def get_jsc_toy(info):
    # TODO: Tanh is not supported by mase yet
    return JSC_Toy(info)


def get_jsc_tiny(info):
    return JSC_Tiny(info)


def get_jsc_s(info):
    return JSC_S(info)


def get_jsc_ziyun_lab1(info):
    return JSC_ziyun_lab1(info)


def get_jsc_three_layer(info):
    return JSC_Three_Linear_Layers(info)