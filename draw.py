from scipy import optimize
import random
import subprocess
import scipy.io as sio
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import operator
import copy
from scipy import optimize
from randomforest import rf,cluster
from mpl_toolkits.mplot3d import Axes3D
from DNGR import arg_as_list
def f_1(x, A, B):
    return A * x + B



def f_2(x, A, B, C):
    return A * x * x + B * x + C



def f_3(x, A, B, C, D):
    return A * x * x * x + B * x * x + C * x + D
def f_4(x,A):
    return float(A)/x
def draw():
    md = open('res_wine.txt', 'r')
    l=[]
    for i in range(4):
        line=md.readline()
        l.append(line.strip('\n'))
    fig=plt.figure()
    plt.subplot(1,1, 1)
    print l[0]
    print l[1]
    print l[2]
    print l[3]

    l1=arg_as_list(l[0])
    l2 = arg_as_list(l[1])
    l3= arg_as_list(l[2])
    l4 = arg_as_list(l[3])

    la=[]
    lb=[]
    lc=[]
    ld=[]
    s = 3
    for i in range(s):
        if l4[i]>=0.7:
            print l1[i],l4[i]
        if l2[i]<1:
            la.append(l1[i]/float(s+2))
            lb.append(l2[i])
            #lc.append(l3[i])
            ld.append(l4[i])
    l1=la
    l2=lb
    l3=lc
    l4=ld

    print l1,l2
    a= optimize.curve_fit(f_4, l1, l2)[0]
    x1 = np.arange(0, 1, 0.01)
    y1 = f_4(x1,a)
    print a
    #plt.ylim(0.000125, 0.0002)
    plt.scatter(l1, l2)
    #plt.plot(x1, y1, 'green',label='avge')
    plt.show()
    '''fig = plt.figure()
    a, b, c = optimize.curve_fit(f_2, l1, l3)[0]
    x1 = np.arange(2, 175, 0.01)
    y2 = a * x1 * x1 + b * x1 + c
    plt.plot(x1, y2, 'red', label='oob')
    plt.scatter(l1, l3)
    plt.show()'''
    fig = plt.figure()

    a, b, c,d = optimize.curve_fit(f_3, l1, l4)[0]
    x1 = np.arange(2, 175, 0.01)
    y3 = a * x1 * x1*x1 + b * x1 *x1+ c*x1+d
    #plt.plot(x1, y3, 'yellow', label='nmi')
    plt.scatter(l1, l4)
    plt.title('nmi')
    plt.show()
    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l1,l2,l3)
    #ax.plot_trisurf(l1, l2, l3)
    #ax.plot_surface(x1, y1, y2, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l1,l2,l4)
    #ax.plot_trisurf(l1,l2,l4)
    #ax.plot_surface(x1, y1, y3, rstride=1, cstride=1, cmap='rainbow')'''

    plt.show()
draw()