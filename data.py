import os
import os.path as osp
import pickle
import numpy as np
import itertools
import scipy.sparse as sp
import urllib  # import urllib.request ???
from collections import namedtuple

"""
数据集的读取

cora数据集：
    + train：140
    + valid：500（自定义），最大取值1708-140
    + test：1000
citeseer数据集（graph大小3327？）
    + train：120
    + valid：500（自定义），最大取值3312-120
    + test：985
pubmed数据集
    + train：60
    + valid：500（自定义），最大取值19717-60
    + test：1000
"""

# 命名元组Data
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    # 数据集下载路径
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    # 枚举需要下载的数据名称
    # ['ind.cora.x', 'ind.cora.tx', 'ind.cora.allx', 'ind.cora.y', 'ind.cora.ty', 'ind.cora.ally',
    # 'ind.cora.graph', 'ind.cora.test.index']
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency_dict: 邻接信息，，类型为 dict
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/raw
                缓存数据路径: {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        # 保存数据的根路径
        self.data_root = data_root
        # 生成缓存路径{data_root}/processed_cora.pkl
        save_file = osp.join(self.data_root, "processed_cora.pkl")

        # 当数据的缓存文件存在时，将使用缓存文件
        # 否则将下载、进行处理，并缓存到磁盘
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            # 重构.pkl文件中特定的对象
            self._data = pickle.load(open(save_file, "rb"))
        else:
            # 下载数据并写入磁盘
            self.maybe_download()
            self._data = self.process_data()
            # 缓存为.pkl文件
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)  # 保存结构
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric

        原数据中各符号代表的含义：
            + x：带标签的训练节点的特征向量（子图）
            + tx：测试节点的特征向量（子图）
            + allx：（带标签+不带标签）所有训练节点的特征向量（子图）

            + y：带标签的训练节点的one-hot标签（子图）
            + ty：测试节点的one-hot标签（子图）
            + ally：（带标签+不带标签）所有训练节点（allx中）的one-hot标签（子图）

            + graph：字典{节点索引：[邻居节点的索引（列表）]}（全图）
                + graph和allx、ally的节点顺序一致
            + test_index：测试节点在graph中的索引/指示

        """
        print("Process data ...")
        # 读取本地文件中的数据，保存在变量中以待使用
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filenames]
        # 通过索引index划分数据集
        train_index = np.arange(y.shape[0])  # 第0维是训练节点的个数
        val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 再往后找500个是验证集
        sorted_test_index = sorted(test_index)  # 对测试索引进行从小到大排序（不改变原列表）

        # 将训练节点和测试节点特征进行拼接-->按行拼接，得到【全图】的特征表示x
        x = np.concatenate((allx, tx), axis=0)
        # 将训练节点和测试节点one-hot标签-->按行拼接+按列max，得到【全图】的（数值）标签y
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        # x，y也改变为相应的测试顺序？？？
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        # x的第0维是节点数量
        num_nodes = x.shape[0]
        # 初始化mask向量
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        # 通过索引为mask赋值
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        # 邻接字典（表）
        adjacency_dict = graph

        # 打印数据的信息
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        # 返回命名元祖
        return Data(x=x, y=y, adjacency_dict=adjacency_dict,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def maybe_download(self):
        """
        目录中没有的文件需要进行下载，原始数据路径: {data_root}/raw
        """
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):  # 查看文件是否存在
                self.download_data("{}/{}".format(self.download_url, name), save_path)

    # 貌似没用到
    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)  # 获得路径的名称

        # test_index要特判
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")  # 返回一个test指示元组，long类型
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")  # 载入并重构python对象
            out = out.toarray() if hasattr(out, "toarray") else out  # 元组
            return out

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载（某一文件进行下载）"""

        # 如果不存在此路径，那么先创建目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 根据url路径请求下载
        data = urllib.request.urlopen(url)
        filename = os.path.split(url)[-1]  # 提取到路径的最后一个名字（文件名）

        # 向本地的同名文件中写
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())

        return True


# citeseer数据集
class CiteseerData(object):
    # 数据集下载路径
    # download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    # 枚举需要下载的数据名称
    filenames = ["ind.citeseer.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="citeseer", rebuild=False):
        """Citeseer数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 3312 * 3703，类型为 np.ndarray
            * y: 节点的标签，总共包括6个类别，类型为 np.ndarray
            * adjacency_dict: 邻接信息，大小3327（？？），类型为 dict
            * train_mask: 训练集掩码向量，维度为 3312，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 3312，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 3312，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/raw
                缓存数据路径: {data_root}/processed_citesser.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        # 保存数据的根路径
        self.data_root = data_root
        # 生成缓存路径{data_root}/processed_citeseer.pkl
        save_file = osp.join(self.data_root, "processed_citeseer.pkl")

        # 当数据的缓存文件存在时，将使用缓存文件
        # 否则将下载、进行处理，并缓存到磁盘
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            # 重构.pkl文件中特定的对象
            self._data = pickle.load(open(save_file, "rb"))
        else:
            # 下载数据并写入磁盘
            # self.maybe_download()
            self._data = self.process_data()
            # 缓存为.pkl文件
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)  # 保存结构
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric

        原数据中各符号代表的含义：
            + x：带标签的训练节点的特征向量（子图）
            + tx：测试节点的特征向量（子图）
            + allx：（带标签+不带标签）所有训练节点的特征向量（子图）

            + y：带标签的训练节点的one-hot标签（子图）
            + ty：测试节点的one-hot标签（子图）
            + ally：（带标签+不带标签）所有训练节点（allx中）的one-hot标签（子图）

            + graph：字典{节点索引：[邻居节点的索引（列表）]}（全图）
                + graph和allx、ally的节点顺序一致
            + test_index：测试节点在graph中的索引/指示

        """
        print("Process data ...")
        # 读取本地文件中的数据，保存在变量中以待使用
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filenames]
        # 防止数组越界<3312
        # max_index = ally.shape[0] + ty.shape[0]
        # test_index = test_index[test_index < max_index]

        s = test_index.min()
        t = test_index.max()
        tx_zero = np.zeros(tx.shape[1], dtype=np.float).reshape(1, -1)
        ty_zero = np.zeros(ty.shape[1]).reshape(1, -1)
        for i in range(s, t + 1):
            if i not in test_index:
                arr_i = np.array(i).reshape(1, )
                test_index = np.concatenate((test_index, arr_i), axis=0)
                tx = np.concatenate((tx, tx_zero), axis=0)
                ty = np.concatenate((ty, ty_zero), axis=0)

        # 通过索引index划分数据集
        train_index = np.arange(y.shape[0])  # 第0维是训练节点的个数
        val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 再往后找500个是验证集
        sorted_test_index = sorted(test_index)  # 对测试索引进行从小到大排序（不改变原列表）

        # 将训练节点和测试节点特征进行拼接-->按行拼接，得到【全图】的特征表示x
        x = np.concatenate((allx, tx), axis=0)
        # 将训练节点和测试节点one-hot标签-->按行拼接+按列max，得到【全图】的（数值）标签y
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        # x，y也改变为相应的测试顺序？？？
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        # x的第0维是节点数量
        num_nodes = x.shape[0]
        # 初始化mask向量
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        # 通过索引为mask赋值
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        # 邻接字典（表）
        adjacency_dict = graph
        # for key in list(adjacency_dict):
        #     if key >= max_index:
        #         adjacency_dict.pop(key)
        #     else:
        #         adjacency_dict[key] = [v for k, v in enumerate(adjacency_dict[key]) if v < max_index]

        # 打印数据的信息
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        # 返回命名元祖
        return Data(x=x, y=y, adjacency_dict=adjacency_dict,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)  # 获得路径的名称

        # test_index要特判
        if name == "ind.citeseer.test.index":
            out = np.genfromtxt(path, dtype="int64")  # 返回一个test指示元组，long类型
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")  # 载入并重构python对象
            out = out.toarray() if hasattr(out, "toarray") else out  # 元组
            return out


# pubmed数据集
class PubmedData(object):
    # 数据集下载路径
    # download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    # 枚举需要下载的数据名称
    filenames = ["ind.pubmed.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="pubmed", rebuild=False):
        """Pubmed数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 19717 * 500，类型为 np.ndarray
            * y: 节点的标签，总共包括3个类别，类型为 np.ndarray
            * adjacency_dict: 邻接信息，大小19717，类型为 dict
            * train_mask: 训练集掩码向量，维度为 19717，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 19717，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 19717，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/raw
                缓存数据路径: {data_root}/processed_pubmed.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据

        """
        # 保存数据的根路径
        self.data_root = data_root
        # 生成缓存路径{data_root}/processed_pubmed.pkl
        save_file = osp.join(self.data_root, "processed_pubmed.pkl")

        # 当数据的缓存文件存在时，将使用缓存文件
        # 否则将下载、进行处理，并缓存到磁盘
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            # 重构.pkl文件中特定的对象
            self._data = pickle.load(open(save_file, "rb"))
        else:
            # 下载数据并写入磁盘
            # self.maybe_download()
            self._data = self.process_data()
            # 缓存为.pkl文件
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)  # 保存结构
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric

        原数据中各符号代表的含义：
            + x：带标签的训练节点的特征向量（子图）
            + tx：测试节点的特征向量（子图）
            + allx：（带标签+不带标签）所有训练节点的特征向量（子图）

            + y：带标签的训练节点的one-hot标签（子图）
            + ty：测试节点的one-hot标签（子图）
            + ally：（带标签+不带标签）所有训练节点（allx中）的one-hot标签（子图）

            + graph：字典{节点索引：[邻居节点的索引（列表）]}（全图）
                + graph和allx、ally的节点顺序一致
            + test_index：测试节点在graph中的索引/指示

        """
        print("Process data ...")
        # 读取本地文件中的数据，保存在变量中以待使用
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filenames]
        # 防止数组越界<3312
        test_index = test_index[test_index < (ally.shape[0] + ty.shape[0])]
        # 通过索引index划分数据集
        train_index = np.arange(y.shape[0])  # 第0维是训练节点的个数
        val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 再往后找500个是验证集
        sorted_test_index = sorted(test_index)  # 对测试索引进行从小到大排序（不改变原列表）

        # 将训练节点和测试节点特征进行拼接-->按行拼接，得到【全图】的特征表示x
        x = np.concatenate((allx, tx), axis=0)
        # 将训练节点和测试节点one-hot标签-->按行拼接+按列max，得到【全图】的（数值）标签y
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        # x，y也改变为相应的测试顺序？？？
        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        # x的第0维是节点数量
        num_nodes = x.shape[0]
        # 初始化mask向量
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        # 通过索引为mask赋值
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        # 邻接字典（表）
        adjacency_dict = graph

        # 打印数据的信息
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        # 返回命名元祖
        return Data(x=x, y=y, adjacency_dict=adjacency_dict,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)  # 获得路径的名称

        # test_index要特判
        if name == "ind.pubmed.test.index":
            out = np.genfromtxt(path, dtype="int64")  # 返回一个test指示元组，long类型
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")  # 载入并重构python对象
            out = out.toarray() if hasattr(out, "toarray") else out  # 元组
            return out
