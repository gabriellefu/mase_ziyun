import sys
# sys.path.remove( '/workspace/machop/')
sys.path
sys.path.append("/rds/general/user/zf923/home/mase_ziyun/machop")
import numpy as np
from chop.passes.graph.analysis.flop_estimator.calculator import calc_modules

def flop_pass_report(graph):
    number_flop=0

    for node in graph.fx_graph.nodes:
        module=node.meta['mase'].module
        #print(node.meta['mase'].parameters["common"]["mase_type"])
        if (node.meta['mase'].parameters["common"]["mase_type"] = "placeholder") or (node.meta['mase'].parameters["common"]["mase_type"] = "output"):
            flop_temp=0
        else:
            in_data=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["value"]
            out_data=node.meta['mase'].parameters["common"]["results"]["data_out_0"]["value"]
            flop_temp=calc_modules.calculate_modules(module, [in_data], [out_data])["computations"]   
            #print("flop_temp in this layer=",flop_temp)  
        
        number_flop+=flop_temp
        #print("number_flop=",number_flop)

    return number_flop


def BitOP_pass_report(graph):
    number_flop=0

    for node in graph.fx_graph.nodes:
        module=node.meta['mase'].module
        #print(node.meta['mase'].parameters["common"]["mase_type"])
        if (node.meta['mase'].parameters["common"]["mase_type"] is "placeholder") or (node.meta['mase'].parameters["common"]["mase_type"] is "output"):
            flop_temp=0
        else:
            in_data=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["value"]
            out_data=node.meta['mase'].parameters["common"]["results"]["data_out_0"]["value"]
            flop_temp=calc_modules.calculate_modules(module, [in_data], [out_data])["computations"]   
            #print("flop_temp in this layer=",flop_temp)  
        
        number_flop+=flop_temp
        #print("number_flop=",number_flop)

    return number_flop


def count_bitop