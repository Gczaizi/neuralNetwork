# coding: utf-8

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

# 神经网络类定义
class nerualNetwork:

    # 初始化神经网络
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # 设置神经网络输入层、隐藏层、输出层节点数量
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # 链接权重矩阵
        # self.wih = (np.random.rand(self.hNodes, self.iNodes) - 0.5)
        # self.who = (np.random.rand(self.oNodes, self.hNodes) - 0.5)
        self.wih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        # 设置学习率
        self.lR = learningRate

        # 激活函数是 sigmod 函数
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入转换为矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 隐藏层输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)

        # 计算输出层误差
        output_errors = targets - final_outputs
        # 隐藏层误差按照权重反向传播
        hidden_errors = np.dot(self.who.T, output_errors)

        # 反向传播更新内部权重 wih: 输入层到隐藏层权重矩阵，who: 隐藏层到输出层权重矩阵
        self.who += self.lR * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lR * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    # 查询神经网络
    def query(self, inputs_list):
        # 将输入转换为矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 隐藏层输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



if __name__ == '__main__':

    # 输入层、隐藏层、输出层节点 3
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率 0.3
    learning_rate = 0.1
    # 创建神经网络
    n = nerualNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    train_data_file = open("./data/mnist_train.csv", "r")
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    epochs = 5

    for e in range(epochs):
        for record in train_data_list:
            all_values = record.split(',')

            scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            
            n.train(scaled_input, targets)
            pass
        pass

    test_data_file = open("./data/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        # print(correct_label, "correct label")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outpus = n.query(inputs)
        label = np.argmax(outpus)
        # print(label, "network's answer")
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass
    
    # print(scorecard)
    scorecard_array = np.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)