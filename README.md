# CV-project1

## 训练：参数查找
运行train.py文件以进行参数查找

参数空间为：

学习率 learning_rate: \[0.005, 0.002, 0.001\]

正则化系数 penalty: \[0.01, 0.001\]

中间层神经元个数 neural_num: \[50, 100, 300\]

查找的过程将显示在终端，同时会写入到.\\model路径下的 search_process.txt 文件中

最优模型会被保存为.\\model路径下的 best_net.pkl 文件

## 训练：指定网络参数
若要自行制定参数进行训练，请以以下格式运行train_one_model.py文件

python train_one_model.py --model_name \[保存的模型文件名\] --max_iter \[迭代次数\] --learning_rate \[初始学习率\] --penalty \[正则化系数\] --neural_num \[中间层神经元个数\] --visualize \[训练完成后是否可视化，y是 n否\]

训练好的模型会被保存在.\\model文件夹中，请注意我们的程序会自动在名称最后添加_net

## 测试
若要测试已经训练好的模型，清以以下格式运行test.py文件

python test.py --model_name \[要测试的模型文件名\]

测试的结果会输出在终端中

## 参数可视化
visualize.py文件可以用来查看网络参数的可视化，请按以下命令进行运行

python visualize.py --model_name \[要可视化的模型文件名\]

可视化元素包括：训练时的错误率曲线、loss曲线、训练后网络的两层权重


