# GraphSAGE-Cora-Citeseer-Pubmed
+ 这是GraphSAGE模型在Cora、Citeseer、Pubmed数据集上的复现代码

+ 语言：PyTorch
+ 参考代码：[https://github.com/FighterLYL/GraphNeuralNetwork/tree/master/chapter7](https://github.com/FighterLYL/GraphNeuralNetwork/tree/master/chapter7)

### 文件说明

| 文件/文件夹 | 说明                                                      |
| :---------- | --------------------------------------------------------- |
| main.py     | 基于Cora、Citeseer、Pubmed（可选择）数据集的GraphSage示例 |
| net.py      | 主要是GraphSage定义                                       |
| data.py     | 主要是Cora数据集准备                                      |
| sampling.py | 简单的采样接口                                            |
| cora        | 存放Cora数据集的文件夹                                    |
| citeseer    | 存放Citeseer数据集的文件夹                                |
| pubmed      | 存放Pubmed数据集的文件夹                                  |
| data        | 所有的数据集                                              |

### 运行示例

```shell
cd GraphSAGE-Cora-Citeseer-Pubmed
python main.py
```

TODO: 支持在GPU中运行