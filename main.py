"""
基于Cora的GraphSage示例
"""
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from net import GraphSage
from data import CoraData
from data import CiteseerData
from data import PubmedData
from sampling import multihop_sampling

from collections import namedtuple
import matplotlib.pyplot as plt

dataset = "cora"
assert dataset in ["cora", "citeseer", "pubmed"]
num_layers = 2
assert num_layers in [2, 3]

if dataset == "cora":
    INPUT_DIM = 1433  # 输入维度
    if num_layers == 2:
        # Note: 采样的邻居阶数需要与GCN的层数保持一致
        HIDDEN_DIM = [256, 7]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
        NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数
    else:
        # Note: 采样的邻居阶数需要与GCN的层数保持一致
        HIDDEN_DIM = [256, 128, 7]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
        NUM_NEIGHBORS_LIST = [10, 5, 5]  # 每阶采样邻居的节点数

elif dataset == "citeseer":
    INPUT_DIM = 3703  # 输入维度
    if num_layers == 2:
        # Note: 采样的邻居阶数需要与GCN的层数保持一致
        HIDDEN_DIM = [256, 6]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
        NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数
    else:
        # Note: 采样的邻居阶数需要与GCN的层数保持一致
        HIDDEN_DIM = [256, 128, 6]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
        NUM_NEIGHBORS_LIST = [10, 5, 5]  # 每阶采样邻居的节点数

else:
    INPUT_DIM = 500  # 输入维度
    if num_layers == 2:
        # Note: 采样的邻居阶数需要与GCN的层数保持一致
        HIDDEN_DIM = [256, 3]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
        NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数
    else:
        # Note: 采样的邻居阶数需要与GCN的层数保持一致
        HIDDEN_DIM = [256, 128, 3]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
        NUM_NEIGHBORS_LIST = [10, 5, 5]  # 每阶采样邻居的节点数

# 定义超参数
# INPUT_DIM = 1433  # 输入维度
# # Note: 采样的邻居阶数需要与GCN的层数保持一致
# HIDDEN_DIM = [128, 7]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
# NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数

# INPUT_DIM = 3703  # 输入维度
# # Note: 采样的邻居阶数需要与GCN的层数保持一致
# HIDDEN_DIM = [128, 6]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
# NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数

# INPUT_DIM = 500  # 输入维度
# # Note: 采样的邻居阶数需要与GCN的层数保持一致
# HIDDEN_DIM = [128, 3]  # 隐藏单元节点数（2层模型，最后一个是输出的类别）
# NUM_NEIGHBORS_LIST = [10, 10]  # 每阶采样邻居的节点数

assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BATCH_SIZE = 16  # 批处理大小
EPOCHS = 10
NUM_BATCH_PER_EPOCH = 20  # 每个epoch循环的批次数

if dataset == "citeseer":
    LEARNING_RATE = 0.1  # 学习率
else:
    LEARNING_RATE = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])

# 载入数据
if dataset == "cora":
    data = CoraData().data
elif dataset == "citeseer":
    data = CiteseerData().data
else:
    data = PubmedData().data

# x = data.x
if dataset == "citeseer":
    x = data.x
else:
    x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

# 定义训练、验证、测试集
train_index = np.where(data.train_mask)[0]
train_label = data.y
val_index = np.where(data.val_mask)[0]
test_index = np.where(data.test_mask)[0]

# 实例化模型
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_neighbors_list=NUM_NEIGHBORS_LIST,
                  aggr_neighbor_method="mean",
                  aggr_hidden_method="sum").to(DEVICE)
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)


# 定义训练函数
def train():
    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []

    model.train()  # 训练模式
    for e in range(EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        if e % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1

        for batch in range(NUM_BATCH_PER_EPOCH):  # 每个epoch循环的批次数
            # 随即从训练集中抽取batch_size个节点(batch_size,num_train_node)
            batch_src_index = np.random.choice(train_index, size=(BATCH_SIZE,))
            # 根据训练节点提取其标签(batch_size,num_train_node)
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            # 进行多跳采样(num_layers+1,num_node)
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
            # 根据采样的节点id构造采样节点特征(num_layers+1,num_node,input_dim)
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]

            # 送入模型开始训练
            batch_train_logits = model(batch_sampling_x)
            # 计算损失
            loss = criterion(batch_train_logits, batch_src_label)
            train_loss += loss.item()

            # 更新参数
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新

            # 计算训练精度
            _, pred = torch.max(batch_train_logits, dim=1)
            correct = (pred == batch_src_label).sum().item()
            acc = correct / BATCH_SIZE
            train_acc += acc

            validate_loss, validate_acc = validate()
            val_loss += validate_loss
            val_acc += validate_acc

            print(
                "Epoch {:03d} Batch {:03d} train_loss: {:.4f} train_acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}".format
                (e, batch, loss.item(), acc, validate_loss, validate_acc))

        train_losses.append(train_loss / NUM_BATCH_PER_EPOCH)
        train_acces.append(train_acc / NUM_BATCH_PER_EPOCH)
        val_losses.append(val_loss / NUM_BATCH_PER_EPOCH)
        val_acces.append(val_acc / NUM_BATCH_PER_EPOCH)

        # 测试
        test()

    res_plot(EPOCHS, train_losses, train_acces, val_losses, val_acces)


# 定义测试函数
def validate():
    model.eval()  # 测试模式
    with torch.no_grad():  # 关闭梯度
        val_sampling_result = multihop_sampling(val_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        val_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in val_sampling_result]
        val_logits = model(val_x)
        val_label = torch.from_numpy(data.y[val_index]).long().to(DEVICE)
        loss = criterion(val_logits, val_label)
        predict_y = val_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, val_label).float().mean().item()

        return loss.item(), accuarcy


# 定义测试函数
def test():
    model.eval()  # 测试模式
    with torch.no_grad():  # 关闭梯度
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)


def res_plot(epoch, train_losses, train_acces, val_losses, val_acces):
    epoches = np.arange(0, epoch, 1)
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    # 画出训练结果
    plt.plot(epoches, train_losses, 'b', label='train_loss')
    plt.plot(epoches, train_acces, 'r', label='train_acc')
    # plt.setp(ax.get_xticklabels())
    plt.legend()

    plt.subplot(1, 2, 2, sharey=ax)
    # 画出训练结果
    plt.plot(epoches, val_losses, 'k', label='val_loss')
    plt.plot(epoches, val_acces, 'g', label='val_acc')
    plt.legend()

    plt.savefig('res_plot.jpg')

    plt.show()


# main函数，程序入口
if __name__ == '__main__':
    train()
