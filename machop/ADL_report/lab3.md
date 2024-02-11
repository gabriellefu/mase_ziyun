# Lab 3

1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
* In this task, I apply the latency as the new quality metrics. The code is shown as follows. 
![alt text](lab3_1.png)
*In the code, the lantcy of each iteration is recorded and we used the mean of the recorded lantency as the lantency of the whole network in this each config.
*The result of lantency in each config is shows are follows.
![alt text](lab3_1_result.png)


2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It's important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).


3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.


4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.