#coding:utf-8
from numpy import *
from sklearn import preprocessing
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pylab
#import numpy as np
import itertools
import pandas as pd
import random

#数据读取-单因变量与多因变量
def loadDataSet01(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    row = len(arrayLines)
    x = mat(zeros((row, 12)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split('\t')
        x[index,:] = curLine[0: 12]
        y[index,:] = curLine[-1]
        index +=1
    return x, y

#数据标准化
def stardantDataSet(x0, y0):
    e0 = preprocessing.scale(x0)
    f0 = preprocessing.scale(y0)
    return e0, f0

#求均值-标准差
def data_Mean_Std(x0, y0):
    mean_x = mean(x0, 0)
    mean_y = mean(y0, 0)
    std_x = std(x0, axis=0, ddof=1)
    std_y = std(y0, axis=0, ddof=1)
    return mean_x, mean_y, std_x, std_y

#PLS核心函数
def PLS(x0, y0):
    e0, f0 = stardantDataSet(x0,y0)
    e0 = mat(e0); f0 = mat(f0); m = shape(x0)[1]; ny=shape(y0)[1]
    w = mat(zeros((m, m))).T; w_star = mat(zeros((m, m))).T
    chg = mat(eye((m)))
    my = shape(x0)[0];ss = mat(zeros((m,1))).T
    t = mat(zeros((my,m))); alpha= mat(zeros((m,m)))
    press_i = mat(zeros((1,my)))
    press = mat(zeros((1, m)))
    Q_h2 = mat(zeros((1, m)))
    beta = mat(zeros((1,m))).T
    for i in range(1,m+1):
        #计算w,w*和t的得分向量
        matrix = e0.T * f0 * (f0.T * e0)
        val, vec = linalg.eig(matrix)#求特征向量和特征值
        sort_val = argsort(val)
        index_vec = sort_val[:-2:-1]
        w[:,i-1] = vec[:,index_vec]#求最大特征值对应的特征向量
        w_star[:,i-1] =  chg * w[:,i-1]
        t[:,i-1] = e0 * w[:,i-1]
        #temp_t[:,i-1] = t[:,i-1]
        alpha[:,i-1] = (e0.T * t[:,i-1]) / (t[:,i-1].T * t[:,i-1])
        chg = chg * mat(eye((m)) - w[:,i-1] * alpha[:,i-1].T)
        e = e0 - t[:,i-1] * alpha[:,i-1].T
        e0 = e
        #计算ss(i)的值
        #beta = linalg.inv(t[:,1:i-1], ones((my, 1))) * f0
        #temp_t = hstack((t[:,i-1], ones((my,1))))
        #beta = f0\linalg.inv(temp_t)
        #beta = nnls(temp_t, f0)
        beta[i-1,:] = (t[:,i-1].T * f0) /(t[:,i-1].T * t[:,i-1])
        cancha = f0 - t * beta
        ss[:,i-1] = sum(sum(power(cancha, 2),0),1)#注：对不对？？？
        for j in range(1,my+1):
            if i==1:
                t1 = t[:, i - 1]
            else:
                t1 = t[:,0:i]
            f1=f0
            she_t = t1[j-1,:]; she_f = f1[j-1,:]
            t1=list(t1); f1 = list(f1)
            del t1[j-1];  del f1[j-1] #删除第j-1个观察值
            #t11 = np.matrix(t1)
            #f11 = np.matrix(f1)
            t1 = array(t1); f1 = array(f1)
            if i==1:
                t1 = mat(t1).T; f1 = mat(f1).T
            else:
                t1 = mat(t1); f1 = mat(f1).T

            beta1 = linalg.inv(t1.T * t1) * (t1.T * f1)
            #beta1 = (t1.T * f1) /(t1.T * t1)#error？？？
            cancha = she_f - she_t*beta1
            press_i[:,j-1] = sum(power(cancha,2))
        press[:,i-1]=sum(press_i)
        if i>1:
            Q_h2[:,i-1] =1-press[:,i-1]/ss[:,i-2]
        else:
            Q_h2[:,0]=1
        if Q_h2[:,i-1]<0.0975:
            h = i
            break
    return h, w_star, t, beta

##计算反标准化之后的系数
def Calxishu(xishu, mean_x, mean_y, std_x, std_y):
    n = shape(mean_x)[1]; n1 = shape(mean_y)[1]
    xish = mat(zeros((n, n1)))
    ch0 = mat(zeros((1, n1)))
    for i in range(n1):
        ch0[:, i] = mean_y[:, i] - std_y[:, i] * mean_x / std_x * xishu[:, i]
        xish[:, i] = std_y[0, i] * xishu[:, i] / std_x.T
    return ch0, xish

'''
为了获得较可靠的结果，需测试数据（测验）和训练数据（学习）
--可按照6:4的比例划分数据集,即随机分成训练集与测试集
'''
def splitDataSet(x, y):
    m =shape(x)[0]
    train_sum = int(round(m * 0.6))
    test_sum = m - train_sum
    #利用range()获得样本序列
    randomData = range(0,m)
    randomData = list(randomData)
    #根据样本序列进行分割- random.sample(A,rep)
    train_List = random.sample(randomData, train_sum)
    #获取训练集数据-train
    train_x = x[train_List,: ]
    train_y = y[train_List,: ]
    #获取测试集数据-test
    test_list = []
    for i in randomData:
        if i in train_List:
            continue
        test_list.append(i)
    test_x = x[test_list,:]
    test_y = y[test_list,:]
    return train_x, train_y, test_x, test_y

#主函数
if __name__ == '__main__':
    x0, y0 = loadDataSet01('data/000001.txt')#单因变量与多因变量
    # 随机划分数据集- (十折交叉验证)
    # train_x, train_y, test_x, test_y = splitDataSet(x0, y0)
    #标准化
    e0, f0 = stardantDataSet(x0, y0)
    mean_x, mean_y, std_x, std_y = data_Mean_Std(x0, y0)
    r = corrcoef(x0)
    m = shape(x0)[1]
    n = shape(y0)[1]  # 自变量和因变量个数
    row = shape(x0)[0]
    #PLS函数
    h, w_star, t, beta = PLS(x0, y0)
    xishu = w_star * beta
    #反标准化
    ch0, xish = Calxishu(xishu, mean_x, mean_y, std_x, std_y)

    # 求可决系数和均方根误差
    y_predict = x0 * xish + tile(ch0[0, :], (row, 1))
    y_mean = tile(mean_y, (row, 1))
    SSE = sum(sum(power((y0 - y_predict), 2), 0))
    SST = sum(sum(power((y0 - y_mean), 2), 0))
    SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    RR = SSR / SST
    RMSE = sqrt(SSE / row)
    print ("=============================")
    print (u"h:", h)
    print (u"R2:", RR)
    print (u"RMSE:", RMSE)
    # print u"残差平方和:", SSE
    print (u"回归系数：")
    #print (ch0)
    # xish = list(xish)
    #print (xish)
    print ("=============================")
    tofile = pd.DataFrame(y_predict)
    tofile.to_csv('data/y_predict.csv')