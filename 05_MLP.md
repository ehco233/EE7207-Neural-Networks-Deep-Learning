#### MLP

MLP(Multilayer Feed-Forward Network) 多层前馈神经网络，通常也被称作多层感知机。

主要包含输入层，隐藏层以及输出层。其中隐藏层包含多个计算节点，主要用于学习输入数据的非线性特征。其主要的特点：

- 使用平滑且可微的非线性激活函数，引入非线性，使网络具备非线性的能力
- 包含多个隐藏层，对输入进行非线性特征变换
- 全连接，每一层的每个节点都与下一层的所有节点完全连接

在初始阶段，很重要的是确定neural networks 的架构
1） 输入层的神经元个数，应该与输入变量维度一致
2）输出层的神经元个数，应该与输出变量的维度一致
3）hidden layer 的数量，一般为1-2
4）hidden layer 的神经元数量，自己设置，越复杂的问题，神经元数量越多
   - 第一隐藏层：通常是 输入层神经元个数的 2-3 倍。
   - 后续隐藏层：逐层递减，最终 接近输出层神经元个数。

确定架构之后，需要确定层之间传递的权重：back propagation 算法
- 输出层以及隐藏层的权重需要分别计算
在更新权重的过程中，需要明确：
- learning rate. 太大会不稳定，太小会陷入局部最优。可以考虑加入momentum
- 非线性函数：sigmoid, tanh, relu 等

在训练过程中，可以选择sequential，batch ,minibatch mode三种，主要是确定什么时候更新权重。

实际的训练过程：
1. initialization 权重
2. 确定training samples
3. forward pass 计算输出
4. backward pass 更新权重
5. 迭代直至收敛（目标loss 不在变化，或者已经达到迭代最大值）

备注：
1. 输出与最开始的初始化权重，以及每个epoch reshuffle data影响，每一次结果不一样
2. 太多hidden layer 会 gradient vanish。减少hidden layer 层数或换activation function
3. 越早期梯度下降越小，学习越慢
4.更多的hiddden layer 神经元，捕捉更复杂的关系
