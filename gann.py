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
str1 = 'wine/wine.edgelist'
str2 = 'data/wiki/wiki.edgelist'
str3 = 'data/cora/cora.edgelist'
str4 = 'data/karate/karate.edgelist'
class net:
    def __init__(self,input=178,output=1):
        self.input=input
        self.output=output
        self.hl=random.randint(4,9)
        self.neuron=[]
        self.flag = random.randint(0,2)  # gcn,cnn,dense?
        self.neuron.append(self.flag)#which type?
        self.neuron.append(random.randint(output+3,input-1))
        self.rhl=self.hl
        self.fitness=-2
        self.avge=1
        self.oob=0
        self.nmi=0


        for i in range(2,self.hl):
            if self.neuron[i-1]==output+1:
                self.rhl=i
                break
            if i==self.hl-1:
                self.neuron.append(output+1)
            else:
                if i<3:
                  self.neuron.append(random.randint(output+4-i,self.neuron[i-1]-1))
                else:
                  self.neuron.append(random.randint(output + 1, self.neuron[i - 1] - 1))
    def mute(self):# simple now
        self1=self

        if self1.rhl>4:
            k = random.randint(0, 3)
        else:
            k = random.randint(0, 2)

        if k==0:#change    //likely to select bigger range
         x=0
         l=random.uniform(0,1)
         range=[]
         range.append(self1.input-self1.neuron[2]-2)
         i=2
         length=range[0]+1
         while(i<self1.rhl-1):
             range.append(self1.neuron[i-1]-self1.neuron[i+1]-2)
             length=length+range[i-1]
             i=i+1
         i=1
         range[0]=range[0]/float(length)
         while(i<self1.rhl-2):
             range[i]=range[i]/float(length)
             range[i]=range[i]+range[i-1]
             i=i+1
         i=1
         if l>=0.0 and l<range[0]:
             x=1
         else:
             while(i<self1.rhl-2):
              if(l>=range[i-1] and l<range[i]):
                 x=i+1
                 break
              else:
                  i=i+1
         if x>1 :
            if self.neuron[x+1]<self.neuron[x-1]-1:
             self.neuron[x]=random.randint(self.neuron[x+1]+1,self1.neuron[x-1]-1)
         elif x == 1:
             if self.neuron[x + 1] < self.input - 1:
              self1.neuron[1] = random.randint(self1.neuron[x + 1]+1,self1.input - 1)
         '''else:
            if self.rhl>1 and self.output+1<=self.neuron[x-1]-1:
             self.neuron[x] = random.randint(self.output+1, self.neuron[x-1]-1)'''
        elif k==1 :#add
             x = random.randint(1, self1.rhl - 1)
             #so the first layer is also ok
             if x==1:
                 if self1.input>self1.neuron[1]+1:
                     v = random.randint(self1.neuron[1]+ 1, self1.input - 1)
                     self1.neuron.insert(x, v)
                     self1.rhl = self1.rhl + 1
             else:
              if self1.neuron[x-1]>self1.neuron[x]+1 :
               v=random.randint( self1.neuron[x]+1,self1.neuron[x-1]-1)
               self1.neuron.insert(x,v)
               self1.rhl=self1.rhl+1
        elif k==2:
            m1=(self.flag+1)%3
            m2=(self.flag-1+3)%3
            l=random.uniform(0,1)
            if(l>0.5):
                self.flag=m1
                self.neuron[0]=m1
            else:
                self.flag=m2
                self.neuron[0]=m2
        else:#delete
            if self1.rhl>4:
             x=random.randint(1, self1.rhl - 2)
             v=self1.neuron[x]
             self1.neuron.remove(v)
             self1.rhl=self1.rhl-1
        return self1
    def cross(self,other):
        x = random.randint(2, self.rhl - 1)
        v=self.neuron[x]
        for i in range(1,other.rhl-2):
            if other.neuron[i]> v and other.neuron[i+1]<self.neuron[x-1]:
                l1=other.neuron[0:i+1]
                l2=other.neuron[i+1:]
                l3=self.neuron[0:x]
                l4=self.neuron[x:]
                l1.extend(l4)
                l3.extend(l2)
                if(len(l1)>=4 and len(l3)>=4):
                    self.neuron=l1
                    self.rhl=len(l1)
                    other.neuron=l3
                    other.rhl=len(l3)
                    break;


    def f(self):
        #print '~/Desktop/DNGR-Keras-master/DNGR.py --graph_type ' + 'undirected' + ' --input ' + 'wine.edgelist' + ' --output ' + 'representation'+'--hidden_layers'+str(self.rhl)+'--neurons_hiddenlayer'+'{}'.format(self.neuron)
        #subprocess.call('~/Desktop/DNGR-Keras-master/DNGR.py --graph_type ' + 'undirected' + ' --input ' + 'wine.edgelist' + ' --output ' + 'representation'+'--hidden_layers'+str(self.rhl+'--neurons_hiddenlayer'+'{}'.format(self.neuron),shell=True)
        # f = open('xx', 'w')
        # ouput your data into f
        neuron = open('neorons.txt', 'w')
        print self.neuron
        neuron.write(str(self.neuron))
        neuron.close()
        layer = open('layer.txt', 'w')
        layer.write(str(self.rhl))
        layer.close()
        o='representation'
        o1=o+'.pkl'

        subprocess.call(
            '~/Desktop/e/DNGR.py --graph_type ' + 'undirected' + ' --input ' + str1 + ' --output ' + o,

            shell=True)


        '''df = pd.read_pickle('representation.pkl')
        reprsn = df['embedding'].values
        node_idx = df['node_id'].values'''
        md = open('avge.txt', 'r')
        line = md.readline()
        avge=float(line)
        '''reprsn = [np.asarray(row, dtype='float32') for row in reprsn]
        reprsn = np.array(reprsn, dtype='float32')
        true_labels = [labels[int(node)][0] for node in node_idx]
        true_labels = np.asarray(true_labels, dtype='int32')
        cluster(reprsn, true_labels, n_clusters=3)'''
        #self.avge=rf()
        #fitness=-avge
        #fitness=rf()
        self.nmi=cluster(o1)
        return -avge




def ga():
    po=[]
    size=30
    g=1000
    cmpfun = operator.attrgetter('fitness')
    for k in range(4):#self,cross,mute
     for i in range(size):
         po.append(net(output=i+1))
    for i in range(g):
        print i
        for j in range(size):
            print 'generation        ',i,   ',individual    ', j
            if po[j].fitness==-2:#just for the first time
             po[j].fitness=po[j].f()
            print 'self'
            print j,     po[j].fitness   ,po[j].neuron
            print 'before     ',po[j].neuron
            md1=copy.deepcopy(po[j])#mute
            po[j+size]=md1.mute()#any influence to po[j]?
            print 'after       ',po[j].neuron
            print i,     j+size
            po[j+size].fitness=po[j+size].f()
            print 'mute'
            print j+size,         po[j+size].fitness,    po[j+size].neuron
            md2 = copy.deepcopy(po[j])
            print 'j', po[j].fitness, po[j].neuron
            print 'j+1', po[j+1].fitness, po[j+1].neuron
            if j< size-1:
              md3=copy.deepcopy(po[j+1])
              k=j+1
            elif j==size-1:
                md3 = copy.deepcopy(po[j - 1])
                k=j-1
            print 'before     ', po[j].neuron
            md2.cross(md3)  # any influence to po[j]?
            po[j+3*size]=md2
            po[k+2*size]=md3
            print 'after       ', po[j].neuron
            print i, j + size
            po[j +3* size].fitness = po[j + 3*size].f()
            print 'cross'
            #if po[j+2*size].fitness>=0:
            po[j+2*size].fitness=po[j+2*size].f()

            print 'after       ', po[j].neuron
            print 'cross'
            po[k+2*size].fitness=po[k+2*size].f()
            print 'after       ', po[j].neuron
            print j, po[j].fitness,po[j].nmi, po[j].neuron
            print j + size, po[j + size].fitness,po[j+size].nmi, po[j + size].neuron
            print j + 2*size, po[j + 2*size].fitness,po[j+2*size].nmi   , po[j + 2*size].neuron
            print j + 3*size, po[j + 3*size].fitness,po[j+3*size].nmi, po[j + 3*size].neuron
            if po[j+size].fitness>po[j].fitness:
                po[j]=po[j+size]
            if po[j+2*size].fitness>po[j].fitness:
                po[j]=po[j+2*size]
            if po[j+3*size].fitness>po[j].fitness:
                po[j]=po[j+3*size]
            print j, po[j].fitness, po[j].nmi, po[j].neuron

        #po.sort(key=cmpfun, reverse=True)
        print 'a generation'
        for j in range(size):
            print j   ,  po[j].fitness   ,po[j].nmi    ,po[j].neuron
        l1=[]
        l2=[]
        l3=[]
        l4=[]
        ne = open('neu_wine.txt', 'w')


        for i in range(size):
            l2.append(-po[i].fitness)
            l1.append(po[i].output+1)
            #l3.append(po[i].oob)
            l4.append(po[i].nmi)
            ne.write(str(po[i].neuron)+'\n')
        ne.close()
        print l1
        print l2
        print l3
        #print l4
        c = open('res_wine.txt', 'w')
        c.write(str(l1)+'\n')
        c.write(str(l2)+'\n')
        c.write(str(l3) + '\n')
        c.write(str(l4) )
        c.close()
    '''plt.figure()
    plt.scatter(l1, l2)
    a,b,c=optimize.curve_fit(f_2,l1,l2)[0]
    x1=np.arange(2,100,0.01)
    y1=a*x1*x1+b*x1+c
    plt.plot(x1,y1,'green')
    plt.ylim(0, 0.005)
    plt.show()'''
        #sorted(po,key=lambda x:x.fitness,reverse=True)

    #print str(po[0].neuron)
    # ouput your data into f




'''def dataset1():
    data_mat = np.loadtxt('data/wine/wine_edgelist.txt')
    labels = np.loadtxt('data/wine/wine_category.txt')
    print type(data_mat)
    d = data_mat.tolist()
    data_edge = nx.Graph(data_mat)

    with open('data/wine/wine.edgelist', 'wb') as f:
        nx.write_weighted_edgelist(data_edge, f)
def dataset2():
    data_mat = np.loadtxt('data/wiki/wiki_edgelist.txt')
    labels = np.loadtxt('data/wiki/wiki_category.txt')
    print type(data_mat)
    d = data_mat.tolist()
    data_edge = nx.read_edgelist('data/wiki/wiki_edgelist.txt')

    with open('data/wiki/wiki.edgelist', 'wb') as f:
        nx.write_weighted_edgelist(data_edge, f)
def dataset3():
    data_mat = np.loadtxt('data/cora/cora_edgelist.txt')
    labels = np.loadtxt('data/cora/cora_category.txt')
    print type(data_mat)
    d = data_mat.tolist()
    data_edge = nx.read_edgelist(data_mat)

    with open('data/cora/cora.edgelist', 'wb') as f:
        nx.write_weighted_edgelist(data_edge, f)
dataset1()'''

ga()

