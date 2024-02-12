# Lab 3

1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
>* Latency, model size, the number of FLOPs, and the number of BitOPs can all be considered as metrics.
>* Latency: The time it takes to perform a prediction on a sample.
>* Model size: The number of parameters and memory usage.
>* >* FLOPs: The number of floating-point operations required to perform a prediction.
BitOPs: The number of bitwise operations required to perform a prediction.





2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It's important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).
>* In this task, I apply the latency as the new quality metrics. The code is shown as follows. 
![alt text](lab3_1.png)
>*In the code, the lantcy of each iteration is recorded and we used the mean of the recorded lantency as the lantency of the whole network in this each config.
>*The result of lantency in each config is shows are follows.
![alt text](lab3_1_result.png)
>* This task is a classification task, and  the loss we used is cross-entropy loss, which represents the difference between the predicted and actual labels. Accuracy represents the percentage of correct prediction in classification, therefore they are actually evaluating the same thing.


3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.
>* In this task, I wrote a new toml file in mase/machop/configs/examples/jsc_toy_by_bf.toml and change the samplier to "brute-force".
>* The result of "brute-force" search is shown as follows.

|    | number | software_metrics                    | hardware_metrics                                  | scaled_metrics                               |
|----|--------|--------------------------------------|---------------------------------------------------|----------------------------------------------|
| 0  | 1      | {'loss': 0.847, 'accuracy': 0.717}  | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.717, 'average_bitwidth': 1.6} |
| 1  | 2      | {'loss': 0.882, 'accuracy': 0.673}  | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.673, 'average_bitwidth': 0.8} |
| 2  | 5      | {'loss': 0.979, 'accuracy': 0.625}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.625, 'average_bitwidth': 0.4} |



4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.
>* The result of "tpe" search is shown as follows.

|    | number | software_metrics                   | hardware_metrics                                  | scaled_metrics                              |
|----|--------|--------------------------------------|---------------------------------------------------|---------------------------------------------|
| 0  | 1      | {'loss': 0.94, 'accuracy': 0.674}   | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.674, 'average_bitwidth': 1.6}|
| 1  | 2      | {'loss': 1.002, 'accuracy': 0.656}  | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.656, 'average_bitwidth': 0.8}|
| 2  | 4      | {'loss': 1.08, 'accuracy': 0.583}   | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.583, 'average_bitwidth': 0.4}|


>* Comparing the result of "tpe" search and "brute-force" search, they found the same best performance struture.