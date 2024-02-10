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
        if (node.meta['mase'].parameters["common"]["mase_type"] == "placeholder") or (node.meta['mase'].parameters["common"]["mase_type"] == "output"):
            flop_temp=0
        else:
            type=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["type"]
            in_data=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["value"]
            out_data=node.meta['mase'].parameters["common"]["results"]["data_out_0"]["value"]
            precision=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["precision"][0]
            flop_temp=calc_modules.calculate_modules(module, [in_data], [out_data],type,precision)["flop_computations"] 
            
            # print()
            # print("computataion",calc_modules.calculate_modules(module, [in_data], [out_data],type)["computations"]  )
            # print(node.meta['mase'].parameters["common"]["mase_op"])  
            # print("flop_temp in this layer=",flop_temp)  
        
        number_flop+=flop_temp
        #print("number_flop=",number_flop)

    return number_flop


def BitOP_pass_report(graph):
    number_BitOP=0

    for node in graph.fx_graph.nodes:
        module=node.meta['mase'].module
        #print(node.meta['mase'].parameters["common"]["mase_type"])
        if (node.meta['mase'].parameters["common"]["mase_type"] == "placeholder") or (node.meta['mase'].parameters["common"]["mase_type"] == "output"):
            flop_temp=0
        else:
            type=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["type"]
            in_data=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["value"]
            out_data=node.meta['mase'].parameters["common"]["results"]["data_out_0"]["value"]
            precision=node.meta['mase'].parameters["common"]["args"]["data_in_0"]["precision"][0]
            flop_temp=calc_modules.calculate_modules(module, [in_data], [out_data],type,precision)["bit_computations"]   
            #print("flop_temp in this layer=",flop_temp)  
        
        number_BitOP+=flop_temp
        #print("number_flop=",number_flop)

    return number_BitOP

