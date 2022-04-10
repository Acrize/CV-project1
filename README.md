# CV-project1

## 训练：参数查找
直接运行train.py文件以进行参数查找

参数空间为：

学习率 learning_rate: \[0.005, 0.002, 0.001\]

正则化系数 penalty: \[0.01, 0.001\]

中间层神经元个数 neural_num: \[50, 100, 300\]

查找的过程将显示在终端，同时会写入到.\\model路径下的 search_process.txt 文件中

最优模型会被保存为.\\model路径下的 best_net.pkl 文件
