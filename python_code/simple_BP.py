import numpy as np
import pandas as pd
import random as rd


class BP:
    def __init__(self, layers, neures_array, AcFunc_list, Lambda):
        self.layers = layers + 1 # 包含上输入层
        self.neures_array = neures_array
        self.AcFunc_list = AcFunc_list
        self.Lambda = Lambda


    def initialise(self, data, label,trapro,valpro,process):
        X=self.DataDivider(data,label,trapro,valpro,process)
        n, p = np.shape(X.train)
        # 补全信息
        if len(self.AcFunc_list)==self.layers-1:
            self.neures_array = np.concatenate(([[p]], self.neures_array), axis=1)[0]
            self.AcFunc_list.insert(0, None)

        self.X = X
        self.n = n
        self.Y = self.oneHotEncode(X.tralabel)

        # 初始化数组
        self.W_list = [None]
        for l in range(1, self.layers):
            # locals()['W'+str(l)]=np.random.normal(0,np.sqrt(2/self.neures_array[l-1]),[self.neures_array[l],self.neures_array[l-1]]) # He initiation
            locals()['W' + str(l)] = np.random.uniform(-np.sqrt(6 / self.neures_array[l - 1]),
                                                       np.sqrt(6 / self.neures_array[l - 1]), (
                                                       self.neures_array[l], self.neures_array[l - 1]))  # He initiation
            self.W_list.append(locals()['W' + str(l)])

        self.b_list = [None]
        for l in range(1, self.layers):
            # locals()['b'+str(l)]=np.random.normal(0,np.sqrt(2/self.neures_array[0]),[1,self.neures_array[l]]) # He initiation
            locals()['b' + str(l)] = np.zeros((1, self.neures_array[l]))
            self.b_list.append(locals()['b' + str(l)])

        self.a_list = [X.train]  # 多行存储
        for l in range(1, self.layers):
            locals()['a' + str(l)] = np.zeros([n, self.neures_array[l]])
            self.a_list.append(locals()['a' + str(l)])

        self.delta_list = [None]  # 多行储存
        for l in range(1, self.layers):
            locals()['delta' + str(l)] = np.zeros((1, self.neures_array[l]))
            self.delta_list.append(locals()['delta' + str(l)])

        self.z_list = [None]  # 多行存储
        for l in range(1, self.layers):
            locals()['z' + str(l)] = np.zeros([n, self.neures_array[l]])
            self.z_list.append(locals()['z' + str(l)])
        return True

    def iterate(self, repeatTime, alpha):
        epsi = 1e-7  # 防止1/的地方NAN
        val_acc_list = [0]
        self.nan_tag = 0
        for time in range(repeatTime):
            rand_list = rd.sample(range(self.n), self.n)
            self.a_list[0] = self.a_list[0][rand_list]
            self.Y = self.Y[rand_list]
            # 前馈计算a
            for i in range(self.n):
                for l in range(1, self.layers):
                    self.a_list[l][i] = (self.AcFunc_dict[self.AcFunc_list[l]](self, np.reshape(self.a_list[l - 1][i],[1, self.neures_array[l - 1]]) @ self.W_list[l].T + np.reshape(self.b_list[l], [1, self.neures_array[l]])))

            # 对此样本进行迭代
            for i in range(self.n):
                # 计算各层的净输入
                for l in range(1, self.layers):
                    self.z_list[l][i] = self.a_list[l - 1][i] @ self.W_list[l].T + np.reshape(self.b_list[l],[1,-1])
                # 计算各层的 delta
                self.delta_list[-1] = -(self.AcFunc_dict['d' + self.AcFunc_list[-1]](self, self.z_list[-1][i]) @ np.diag((1 / self.AcFunc_dict[self.AcFunc_list[-1]](self, self.z_list[-1][i] + epsi)).reshape(1,-1)[0]) @ np.reshape(self.Y[i],[-1,1])).T
                for l in range(1, self.layers - 1).__reversed__():
                    self.delta_list[l] = (
                                self.AcFunc_dict['d' + self.AcFunc_list[l]](self, self.z_list[l][i]) @ self.W_list[
                            l + 1].T @ self.delta_list[l + 1].T).T
                # 计算各层参数的导数
                self.dW_list = [None]
                for l in range(1, self.layers):
                    locals()['dW' + str(l)] = np.zeros([self.neures_array[l], self.neures_array[l - 1]])
                    self.dW_list.append(locals()['dW' + str(l)])
                for l in range(1, self.layers):
                    self.dW_list[l] = self.delta_list[l].T @ np.reshape(self.a_list[l - 1][i],[1, self.neures_array[l - 1]]) + self.Lambda * self.W_list[l]

                self.db_list = [None]
                for l in range(1, self.layers):
                    locals()['db' + str(l)] = np.zeros([1, self.neures_array[l]])
                    self.db_list.append(locals()['db' + str(l)])
                for l in range(1, self.layers):
                    self.db_list[l] = self.delta_list[l]
                # 更新各层参数
                for l in range(1, self.layers):
                    self.W_list[l] = self.W_list[l] - alpha * self.dW_list[l]
                    self.b_list[l] = self.b_list[l] - alpha * self.db_list[l]
            val_acc_list.append(self.predict(self.X.valid,self.X.vallabel))
            if val_acc_list[-2]>val_acc_list[-1]:
                break
            if np.isnan(self.loss()):
                self.nan_tag = 1
                break
            self.W_list_backup = copy.deepcopy(self.W_list)
            self.b_list_backup = copy.deepcopy(self.b_list)
        return True

    def predict(self, test, teslabel):
        test_a = [test]
        ntest = np.shape(test)[0]
        if self.nan_tag == 1:
            self.W_list = self.W_list_backup
            self.b_list = self.b_list_backup
        for l in range(1, self.layers):
            locals()['a_test' + str(l)] = np.zeros([ntest, self.neures_array[l]])
            test_a.append(locals()['a_test' + str(l)])
        for i in range(len(teslabel)):
            for l in range(1, self.layers):
                test_a[l][i] = self.AcFunc_dict[self.AcFunc_list[l]](self, np.reshape(test_a[l - 1][i],[1, self.neures_array[l - 1]]) @self.W_list[l].T + np.reshape(self.b_list[l], [1,self.neures_array[l]]))
        testY = test_a[-1]
        prelabel = np.argmax(testY.T, axis=0)
        accuracy = np.sum(prelabel == teslabel) / len(teslabel)
        return accuracy

    def loss(self):
        return -np.trace(self.Y @ self.a_list[-1].T) / np.shape(self.Y)[0]

    def oneHotEncode(self, label):  # 独热编码矩阵,n*c
        if len(np.shape(label)) == 2:
            label = np.reshape(label, [1, np.shape(label)[0] * np.shape(label)[1]])[0]
        c = len(np.unique(label))
        n = max(np.shape(label))
        Y = np.zeros((n, c))
        # Y[range(0,n),int(label)]=1
        for i in range(0, n):
            Y[i, int(label[i])] = 1
        return np.array(Y, dtype=np.float64)

    def Softmax(self, x):
        # if len(np.shape(x)) == 2 and np.min(np.shape(x)) == 1:
        #     x = x.reshape(1,-1)[0]
        #     return np.exp(x) / np.sum(np.exp(x), axis=0)
        # elif len(np.shape(x)) == 1:
        #     return np.exp(x) / np.sum(np.exp(x), axis=0)
        x=x.reshape(-1,1)
        return (np.exp(x) / np.sum(np.exp(x), axis=0)).reshape(1,-1)[0]


    def dSoftmax(self, x):
        x = np.array(x, dtype=np.float64)
        return np.diag(self.Softmax(x).reshape(1,-1)[0]) - self.Softmax(x).reshape(-1,1) @ self.Softmax(x).reshape(1,-1)

    def ReLU(self, x):
        x.reshape(-1,1)
        return np.maximum(x, 0)

    def dReLU(self, x):
        if len(np.shape(x)) == 2 and max(np.shape(x)) > 1:
            print('ReLU function just receive vector but not matrix!')
            return False
        x=np.reshape(x,[1,-1])[0]
        x[x > 0] = 1
        x[x <= 0] = 0
        return np.diag(x)

    def zscore(self,data):
        # data是行向量矩阵
        data = np.array(data, dtype=np.float64)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data - data_mean) @ np.linalg.inv(np.diag(data_std))
        return data

    def DataDivider(self, data, label, trapro, valpro, process):
        '''
        :param data: 行放样本
        :param label: 列标签
        :param trapro: 训练集比例
        :param valpro: 训练集中验证集比例
        :return: X.train行向量训练矩阵, X.tralabel无嵌套的行向量 X.test行向量测试矩阵 X.teslabel无嵌套的行向量
        '''
        class X_struct:
            def __init__(self, Xtrain, tralabel, Xtest, teslabel, Xvalid, vallabel):
                self.train = Xtrain
                self.test = Xtest
                self.valid= Xvalid
                self.tralabel = tralabel
                self.teslabel = teslabel
                self.vallabel= vallabel
        label = label.reshape(1, -1)[0]
        n, p = np.shape(data)
        c = len(np.unique(label))
        train = np.empty([0, p], float)
        tralabel = np.empty([1, 0])
        test = np.empty([0, p], float)
        teslabel = np.empty([1, 0])
        valid = np.empty([0, p], float)
        vallabel = np.empty([1, 0])
        for i in range(0, c):
            ni = np.sum(label == i)
            tranum = int(ni * trapro)
            rand_list = np.array(rd.sample(range(0, ni), ni))
            data_i = data[label == i] # 第i类的数据
            train_part=data_i[rand_list[0:tranum], :] # 训练集部分的数据
            valnum=int(tranum*valpro) # 验证样本个数
            inner_rand_list=np.array(rd.sample(range(0, tranum), tranum)) # 训练集内部随机数列

            valid=np.append(valid,train_part[inner_rand_list[0:valnum],:],axis=0)
            vallabel=np.append(vallabel,i*np.ones([1,valnum]),axis=1)

            train = np.append(train, train_part[inner_rand_list[valnum:], :], axis=0)
            tralabel = np.append(tralabel, i * np.ones([1, tranum-valnum]), axis=1)

            test = np.append(test, data_i[rand_list[tranum:], :], axis=0)
            teslabel = np.append(teslabel, i * np.ones([1, ni - tranum]), axis=1)
        tralabel = tralabel.reshape(1, -1)[0]
        teslabel = teslabel.reshape(1, -1)[0]
        vallabel = vallabel.reshape(1, -1)[0]
        if process=='zscore':
            train=self.zscore(train)
            test=self.zscore(test)
            valid=self.zscore(valid)
        X = X_struct(train, tralabel, test, teslabel, valid, vallabel)
        return X

    AcFunc_dict = {'ReLU': ReLU, 'dReLU': dReLU, 'Softmax': Softmax, 'dSoftmax': dSoftmax}

# ===================================================================
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

