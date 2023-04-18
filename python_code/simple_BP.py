import numpy as np
import pandas as pd
import random as rd


class BP:
    def __init__(self, layers, neures_array, AcFunc_list, Lambda):
        self.layers = layers + 1 # add input layer
        self.neures_array = neures_array
        self.AcFunc_list = AcFunc_list
        self.Lambda = Lambda


    def initialise(self, data, label,trapro,valpro,process):
        X=self.DataDivider(data,label,trapro,valpro,process)
        n, p = np.shape(X.train)
        # to perfect the information
        if len(self.AcFunc_list)==self.layers-1:
            self.neures_array = np.concatenate(([[p]], self.neures_array), axis=1)[0]
            self.AcFunc_list.insert(0, None)

        self.X = X
        self.n = n
        self.Y = self.oneHotEncode(X.tralabel)

        # initialize the array
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

        self.a_list = [X.train]  # store with multi-row array
        for l in range(1, self.layers):
            locals()['a' + str(l)] = np.zeros([n, self.neures_array[l]])
            self.a_list.append(locals()['a' + str(l)])

        self.delta_list = [None]  # store with multi-row array
        for l in range(1, self.layers):
            locals()['delta' + str(l)] = np.zeros((1, self.neures_array[l]))
            self.delta_list.append(locals()['delta' + str(l)])

        self.z_list = [None]  # store with multi-row array
        for l in range(1, self.layers):
            locals()['z' + str(l)] = np.zeros([n, self.neures_array[l]])
            self.z_list.append(locals()['z' + str(l)])
        return True

    def iterate(self, repeatTime, alpha):
        epsi = 1e-7  # avoid the nan with dividing 0
        val_acc_list = [0]
        self.nan_tag = 0
        for time in range(repeatTime):
            rand_list = rd.sample(range(self.n), self.n)
            self.a_list[0] = self.a_list[0][rand_list]
            self.Y = self.Y[rand_list]
            # Feed-forward Calculation a
            for i in range(self.n):
                for l in range(1, self.layers):
                    self.a_list[l][i] = (self.AcFunc_dict[self.AcFunc_list[l]](self, np.reshape(self.a_list[l - 1][i],[1, self.neures_array[l - 1]]) @ self.W_list[l].T + np.reshape(self.b_list[l], [1, self.neures_array[l]])))

            # iterate
            for i in range(self.n):
                # calculate the pure input of each layer of the i-th sample
                for l in range(1, self.layers):
                    self.z_list[l][i] = self.a_list[l - 1][i] @ self.W_list[l].T + np.reshape(self.b_list[l],[1,-1])
                # calculate the delta of each layer
                self.delta_list[-1] = -(self.AcFunc_dict['d' + self.AcFunc_list[-1]](self, self.z_list[-1][i]) @ np.diag((1 / self.AcFunc_dict[self.AcFunc_list[-1]](self, self.z_list[-1][i] + epsi)).reshape(1,-1)[0]) @ np.reshape(self.Y[i],[-1,1])).T
                for l in range(1, self.layers - 1).__reversed__():
                    self.delta_list[l] = (
                                self.AcFunc_dict['d' + self.AcFunc_list[l]](self, self.z_list[l][i]) @ self.W_list[
                            l + 1].T @ self.delta_list[l + 1].T).T
                # calculate the derivatives of each layer
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
                # update the parameters on each layer
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

    def oneHotEncode(self, label):  # one-hot encoding matrix with size of n*c
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
        # data is row sample matrix
        data = np.array(data, dtype=np.float64)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data - data_mean) @ np.linalg.inv(np.diag(data_std))
        return data

    def DataDivider(self, data, label, trapro, valpro, process):
        '''
        :param data: row-sample matrix
        :param label: class label imformation
        :param trapro: the proportion of training set
        :param valpro: the proportion of valid set of the train set
        :return: X.train, X.tralabel, X.test, X.teslabel, X.valid, X.vallabel
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
            data_i = data[label == i] # the samples of the i-th class
            train_part=data_i[rand_list[0:tranum], :] # the samples of training set
            valnum=int(tranum*valpro) # the number of valid samples
            inner_rand_list=np.array(rd.sample(range(0, tranum), tranum)) # Random sequence within the training set

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



