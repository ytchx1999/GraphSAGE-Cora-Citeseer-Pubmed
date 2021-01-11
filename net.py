import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

"""
定义聚合器、更新器、GraphSAGE层
"""


# 聚合
class NeighborAggregator(nn.Module):  # 继承了nn.Module，具有可学习的参数
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """聚合节点邻居

        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})--3种可选择的方式，不包括paper中的LSTM方法
        """
        super(NeighborAggregator, self).__init__()  # 继承父类的方法、属性
        # 形参初始化
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        # 权重参数矩阵(input_dim, output_dim)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))  # Tensor默认为float类型
        # 是否使用bias(output_dim,)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化参数
        """
        # weight初始化为均匀分布（何凯明ImageNet）
        init.kaiming_uniform_(self.weight)
        # bias初始化为0
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        """
        前向传播

        Args:
            neighbor_feature:需要聚合的邻居节点特征(num_src,num_neigh,input_dim)--(源节点数量,邻居数量,input_dim)

        Returns:
            neighbor_hidden:聚合后的消息，用于更新节点的嵌入表示(num_src,output_dim)

        """
        # 选择聚合器的模式，aggr_neighbor(num_src,input_dim)是邻居聚合的表示
        # (num_src,num_neigh,input_dim)-->(num_src,input_dim)
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)  # 对第1维num_neigh进行聚合
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor, _ = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))

        # 矩阵乘法(num_src,input_dim)x(input_dim,output_dim)=(num_src,output_dim)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        # 和bias相加
        if self.use_bias:
            neighbor_hidden += self.bias
        # 返回消息
        return neighbor_hidden  # (num_src,output_dim)

    def extra_repr(self):
        """
        Returns:返回参数字符串
        """
        return 'in_features={}, out_features={}, aggr_method={}'.format(self.input_dim, self.output_dim,
                                                                        self.aggr_method)


# 更新
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, aggr_neighbor_method="mean", aggr_hidden_method="sum"):
        """SageGCN层定义

        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，（就是聚合器的output_dim）
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        # 初始化参数
        assert aggr_neighbor_method in ["mean", "sum", "max"]  # 3种聚合方法
        assert aggr_hidden_method in ["sum", "concat"]  # 2种更新方法

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        # 定义聚合器aggregator
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        # 初始化更新的权重参数（无bias）
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化权重矩阵
        """
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        """
        前向传播

        Args:
            src_node_features:源节点特征(num_src,input_dim)
            neighbor_node_features:需要聚合的邻居节点特征(num_src,num_neigh,input_dim)

        Returns:

        """
        # 聚合操作
        # (num_src,num_neigh,input_dim)-->(num_src,hidden_dim)
        neighbor_hidden = self.aggregator(neighbor_node_features)
        # 线性变换，矩阵乘
        # (num_src,input_dim)x(input_dim,hidden_dim)=(num_src,hidden_dim)
        self_hidden = torch.matmul(src_node_features, self.weight)

        # 更新操作
        if self.aggr_hidden_method == "sum":  # 加法
            # (num_src,hidden_dim)-->(num_src,hidden_dim)
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":  # 拼接
            # (num_src,hidden_dim)-->(num_src,2 * hidden_dim)
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        # L2规范化
        # hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        # 激活
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        """
        Returns:返回参数信息
        """
        # 当aggr_hidden_method = concat, 输出维度为hidden_dim * 2
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        # 返回
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(self.input_dim, output_dim,
                                                                               self.aggr_hidden_method)


# 模型的定义
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors_list, aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """
        GraphSAGE模型的定义

        Args:
            input_dim:源节点的维度
            hidden_dim:每一层的隐藏（输出）维度列表(num_layers,)
            num_neighbors_list:节点0阶，1阶，2阶...采样的邻居数量(num_layers,)
        """
        super(GraphSage, self).__init__()
        # 初始化参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        # 网络的层数就是列表的长度
        self.num_layers = len(num_neighbors_list)

        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method

        # 定义gcn层
        self.gcn = nn.ModuleList()  # 需要自己定义forward函数
        # 第0层是源节点的特征维度-->hidden[0]
        self.gcn.append(SageGCN(input_dim, hidden_dim[0], aggr_neighbor_method=aggr_neighbor_method,
                                aggr_hidden_method=aggr_hidden_method))
        for index in range(0, len(hidden_dim) - 2):
            if self.aggr_hidden_method == "concat":
                hidden_dim[index] *= 2
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1], aggr_neighbor_method=aggr_neighbor_method,
                                    aggr_hidden_method=aggr_hidden_method))
        if self.aggr_hidden_method == "concat":
            hidden_dim[-2] *= 2
        self.gcn.append(
            SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None, aggr_neighbor_method=aggr_neighbor_method,
                    aggr_hidden_method=aggr_hidden_method))  # 最后一层不需要激活

    def forward(self, node_features_list):
        """
        前向传播

        Args:
            node_features_list:节点0阶，1阶，2阶...邻居的列表(num_layers+1,num_node,input_dim)

        Returns:

        """
        hidden = node_features_list  # 采样后的节点k阶邻居特征列表
        # 不同的层，这部分"倒推"求解
        # 虽然是正序，但是最后结果其实是hidden[0]
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            # 每一层模型都需要进行的操作
            for hop in range(self.num_layers - l):
                # (num_src,input_dim)
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                # view改变形状
                # (num_src * num_neigh,input_dim)-->(num_src,num_neigh,input_dim)
                neighbor_node_features = hidden[hop + 1].view((src_node_num, self.num_neighbors_list[hop], -1))
                # 使用源节点和邻居节点进行聚合+更新操作
                h = gcn(src_node_features, neighbor_node_features)
                # 加入到列表中
                next_hidden.append(h)
            # 重新赋值
            hidden = next_hidden
        return hidden[0]  # 最终hidden列表只剩下1个元素(num_src,output_dim)

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(self.input_dim, self.num_neighbors_list)
