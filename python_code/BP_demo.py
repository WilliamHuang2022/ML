import numpy as np
from simple_BP import *
import pandas as pd

table = np.array(pd.read_csv("iris.data", sep=',', header=None))
data = table[:, 0:-1]
n = np.shape(data)[0]
label_str = table[:, -1]
label = -np.ones([n, 1])
label[label_str == 'Iris-setosa'] = 0
label[label_str == 'Iris-versicolor'] = 1
label[label_str == 'Iris-virginica'] = 2
label = label.reshape(1,-1)

trapro = 0.8
valpro=0.2/trapro
process='zscore'
Lambda = 0.1
maxloop = 10000  # 迭代次数
expe_time = 2
alpha_list = np.arange(6,7) * (1e-3) # 0.006可以达到100%
bp1 = BP(2, np.array([[4, 3]]), ['ReLU', 'Softmax'], Lambda)


acc_table = np.zeros([len(alpha_list), expe_time])
print(alpha_list)
for i in range(expe_time):
    for al in range(len(alpha_list)):
        alpha = alpha_list[al]
        bp1.initialise(data, label, trapro, valpro, process)
        bp1.iterate(maxloop, alpha)
        acc_table[al, i] = bp1.predict(bp1.X.test, bp1.X.teslabel)

print(acc_table)
