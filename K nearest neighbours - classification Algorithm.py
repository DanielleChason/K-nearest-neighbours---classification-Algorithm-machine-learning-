
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def knn(set_images, labels, query, k):
     dis = []
     for i in range(len(set_images)):
         #calc dictance for each image in train:
         distance = numpy.linalg.norm(set_images[i]-query)
         ##add tupple (distance, label):
         dis.append((distance, labels[i]))
     dis = sorted(dis)   #sort by distances
     dis = dis[:k]   #k nearest images
     k_near = [item[1] for item in dis]

     dic = {}
     cnt=0
     max=''
     for i in reversed(k_near):
        dic[i] = dic.get(i, 0) + 1
        if (dic[i] >= cnt):
            cnt=dic[i]
            max=i

     return(max)

def accuracy(n, k):
    #get n from training set:
    s=train[:n]
    s_lables=train_labels[:n]
    cnt=0
    for i in range (len(test)):
        p=knn(s, s_lables, test[i], k)
        if (p==test_labels[i]):
            cnt+=1

    res=cnt/len(test)
    return res



#tests:

def graph1():
    x_axis = [i for i in range(1, 101)]
    y_axis= []
    for k in range(1, 101):
        y_axis.append(accuracy(1000, k))
        print (k)

    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.plot(x_axis, y_axis)
    plt.show()

def graph2():
    x_axis = [i for i in range(100, 5001, 100)]
    y_axis = []
    for n in range(100, 5001, 100):
        y_axis.append(accuracy(n, 1))

    plt.xlabel("n")
    plt.ylabel("accuracy")
    plt.plot(x_axis, y_axis)
    plt.show()


