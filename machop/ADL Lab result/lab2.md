1.	Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr ... You might find the doc of torch.fx useful.


The functionality of report_graph_analysis_pass is the report the graph of the model we used.


placeholder represents a function input. 

get_attr retrieves a parameter from the module hierarchy. 

call_function applies a free function to some values.


output contains the output of the traced function in its args[0] attribute.

2.	What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?


3.	Explain why only 1 OP is changed after the quantize_transform_pass.
The quantize_transform_pass we use is to change the linear 



4.	Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize_quantization_analysis_pass useful.

![Alt text](image.png)
5.	Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the pass_args for your custom network might be different if you have used more than the Linear layer in your network.
![Alt text](image-2.png)

6.	Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers .、

![Alt text](image-1.png)


7.Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

Optional: Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).