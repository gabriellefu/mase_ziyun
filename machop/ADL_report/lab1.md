# Lab 1
1. What is the impact of varying batch sizes and why?
>* In this task, the max-epochs is set as 10 and learning-rate is set as 1e-3. The results is shown as follows.

| Batch Size | Test Accuracy | Test Loss |
|:----------:|:-------------:|:---------:|
| 512        | 0.599259      | 1.036578  |
| 256        | 0.716496      | 0.846329  |
| 128        | 0.717215      | 0.852131  |
| 64         | 0.597643      | 1.057428  |
| 32         | 0.680349      | 1.073670  |
>* As it's shown in the table, the test accuracy has the best performance when the batch size is 128 and 256, and neither too high or too low of batch size would decrease the performacne of the network. 

2. What is the impact of varying maximum epoch number?
>* In this task, the learning rate is set as default and the batch size is set as 256. The result is shown as follows.

|Max Epochs| Test Accuracy | Test Loss |
|:----------:|:-------------:|:---------:|
| 20         | 0.585559         | 1.255687      |
| 15         | 0.530963         | 1.281292      |
| 10         | 0.716496         | 0.846329      |
| 5          | 0.466995         | 1.413420      |
| 1          | 0.287315         | 1.531749      |
>* As it's shown in the result, the performance is the best when max epoch is 10, and too many or too less epoches would make the result worse.

3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
>* The batch size is set as 256 and max epoch is 10 in this task.

|Learning Rate| Test Accuracy | Test Loss |
|:----------:|:-------------:|:---------:|
| 1e-10  | 0.585559      | 1.255687  |
| 1e-6      | 0.530963      | 1.281292  |
| 1e-3         | 0.716496      | 0.846329  |
| 1e-2          | 0.466995      | 1.413420  |
| 1e-1          | 0.570196      | 1.199834  |


>* The result shows that the accuracy is the highest when learning rate is 1e-3. When the learning rate is small, the model is not finish leanring and when the learning rate is too big, the model is oscillated and the result may not be good as well.

4. Implement a network that has in total around 10x more parameters than the toy network.
>* In this task, I set a network named jsc-ziyun-lab1 with the following structure.
```python
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
            nn.Linear(32, 64),  #
            nn.BatchNorm1d(64),  #
            nn.ReLU(64),  # 
            nn.Linear(64, 32),  # 
            nn.BatchNorm1d(32),  # 
            nn.ReLU(32),  # 
            nn.Linear(32, 12),  # 
            nn.BatchNorm1d(12),  # 
            nn.ReLU(12),  # 
            nn.Linear(12, 8),  # 
            nn.BatchNorm1d(8),  # 
            nn.ReLU(8),  # 
            nn.Linear(8, 5),  # 
            nn.BatchNorm1d(5),  # 
            nn.ReLU(5),
        )
    def forward(self, x):
        return self.seq_blocks(x)
```
>* This ntework has a parameter number of 5.7 K.

5. Test your implementation and evaluate its performance.
>* The performance of the new network is shown in the table.
>* Becasue the network is much more larger than jsc-tiny nodel, the best performance learning rate is 1e-2 which can be higher than jsc-tiny model.
>* The average accuracy of this network is also higher than jsc-tiny.

| Learning Rate | Max Epochs | Batch Size | Test Accuracy Epoch   | Test Loss Epoch     |
|---------------|------------|------------|-----------------------|---------------------|
| 1e-3          | 10         | 64         | 0.7431572675704956    | 0.7199923396110535  |
| 1e-3          | 10         | 128        | 0.7491360902786255    | 0.6964259147644043  |
| 1e-2          | 1          | 256        | 0.7213901281356812    | 0.7569963335990906  |
| 1e-2          | 10         | 256        | 0.7516644597053528    | 0.6871607899665833  |
| 1e-2          | 10         | 512        | 0.7513148188591003    | 0.6859774589538574  |
| 1e-2          | 15         | 256        | 0.7518620491027832    | 0.6861042380332947  |
| 1e-2          | 15         | 512        | 0.7526625990867615    | 0.6838246583938599  |
| 1e-2          | 20         | 512        | 0.7531135678291321    | 0.6822382211685181  |
| 1e-1          | 10         | 256        | 0.7458933591842651    | 0.7063556909561157  |
| 1             | 10         | 128        | 0.35242345929145813   | 1.2427880764007568  |
| 1             | 10         | 1024       | 0.35237279534339905   | 1.2416390180587769  |
