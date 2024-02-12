# Lab 2
1. Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr ... 



>* The result of function report_graph_analysis_pass on the model we trained in lab1 is shown as follows.
![alt text](lab2_1.png)
>* As it's shown in the result, the function report the graph generated on the network, the architecture of the tiny network, and the type and hyper parameters in each layers.
>* The meanings of the printed jargons are shown as follows.
>>+ "Placeholder" repersent the input of the graph.
>> + "Get_attr" retrieves a parameter from the module hierarchy.
>>+ "Call_function" refers to the number of nodes that apply functions like add or sum on some values.
>>+ "Call_method" refers to the number of nodes applying a method of the node like relu().
>>+ "Call_module" mears the number of nodes representing a call to modules, usually a custom and complicated one.
>>+ "Output" contains the output of the traced function in its args[0] attribute.



<br>

2.	What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?
>* The functionality of "profile_statistics_analysis_pass" is to retrieve the weight statistics and  act statistics in each node of the graph and profile them.
>* The trucated result of report_node_meta_param_analysis_pass on the "jsc-tiny" model we train in lab1 is shown as follows.
![alt text](LAB1_2.png)
As we can refer from the graph, "report_node_meta_param_analysis_pass" present the detailed statistics of each node in the graph of the network including the "min", "max", "count" and "range" of the input of a "relu" layer, and the "count","variance", and "mean" of the weight of a linear layer.

<br>

3.	Explain why only 1 OP is changed after the quantize_transform_pass.
>* In the above task, we found out the architecture of the "jsc-tiny" model which concludes only one linear node. The influence of the pass we used is to quantize the linear node and change its datatype into "integer" with the precision we set. Since there is only one linear node in the gragh, there's inly 1 OP changed after the quantize_transform_pass.

<br>

4.	Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize_quantization_analysis_pass useful.
>* The code I wrote to traverse both mg and ori_mg is shown as follows.
![alt text](lab2_4_code.png)
>*The result of the code is shown as follows. As we can see from the output, the data type of the "linear" node has changed from original float into integer as we set in the pass.
![alt text](lab2_4_result.png)

<br>

5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.

>* Following the same steps as we do to qutantize the jsc-tiny network. The result of function summarize_quantization_analysis_pass() on this model is shown as follows.
![alt text](lab2_5.png)

>* Because there are multiple "Linear" layer in the model I set, there are 7 changes in total in the model.
<br>

6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the [Quantized Layers](../../machop/chop/passes/transforms/quantize/quantized_modules/linear.py) .
![alt text](lab2_6.png)

>* The code and the output is shown as follows. As we can see from the output, with the same input, the output of the model of mg and the model of mg_ori are different, which indicates that the the weights of these layers are indeed quantised.

<br>

7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.
>* We loaded the model we pre-train in the configs/examples/jsc_toy_by_type.toml file.
>* We enter "./ch transform --config configs/examples/jsc_toy_by_type.toml --task cls --cpu=0" in the termial.
>* The result is shown as follows.

| Original type   | OP           | Total | Changed | Unchanged |
|-----------------|--------------|-------|---------|-----------|
| BatchNorm1d     | batch_norm1d | 1     | 0       | 1         |
| Linear          | linear       | 1     | 1       | 0         |
| ReLU            | relu         | 2     | 0       | 2         |
| output          | output       | 1     | 0       | 1         |
| x               | placeholder  | 1     | 0       | 1         |



<br>


8. (Optional) Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).

>* For the tasks, I wrote the following code which traverse the graph and adding the computation of each nodes. I used the function calculate_modules() and changed the code in the following places.
```python
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
```
>* Get the node.meta['mase'].parameters["common"]["args"]["data_in_0"]["type"] of each node and transmit it to calculate_modules().
>*  Determine the computation is floating-point operations or not with this "type" parameters.
>* The FLOP of ori_mg and mg of jsc-tiny model are 1320 and 680 respectively.
>* The code of BitOP calculation is shown as follows.
>* Just like the FLOP calculation, I used the calculate_modules() function.
```python
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
```
>* Get the node.meta['mase'].parameters["common"]["args"]["data_in_0"]["precision"][0] of each node to gain how many bits it takes and transmit this value to calculate_modules(), and multiple this value to it's computation.
>* Our result shows that the BitOP of ori_mg and mg of the jsc-tiny model are 42240 and 26880 respectively.

