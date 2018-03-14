#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, noise,Conv1D,Flatten,Reshape
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import networkx as nx
import pandas as pd
import argparse
from utils import DataGenerator
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import logging
import sys
import pdb
import subprocess
import ast
import os
import tensorflow as tf
from gcn1 import GraphConvolution
import matplotlib.pyplot as plt
str1='data/wine/wine_edgelist.txt'
str2='data/wiki/wiki_edgelist.txt'
str3='data/cora/cora_edgelist.txt'
str4='data/karate/karate_edgelist.txt'
def read_graph(filename,g_type):
	with open(filename,'rb') as f:
		if g_type == "undirected":
			G = nx.read_weighted_edgelist(f)
  		else:
			G = nx.read_weighted_edgelist(f,create_using=nx.DiGraph())#directed graph
		node_idx = G.nodes()
	adj_matrix = np.asarray(nx.adjacency_matrix(G, nodelist=None,weight='weight').todense())
	#print adj_matrix
	#np.savetxt(str1,adj_matrix,fmt='%f')
	#print node_idx
	return adj_matrix,node_idx


def scale_sim_mat(mat):
	# Scale Matrix by row
	mat  = mat - np.diag(np.diag(mat))
	D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0)))
	mat = np.dot(D_inv,  mat)

	return mat

def random_surfing(adj_matrix,max_step,alpha):
	# Random Surfing
	nm_nodes = len(adj_matrix)
	adj_matrix = scale_sim_mat(adj_matrix)
	P0 = np.eye(nm_nodes, dtype='float32')
	M = np.zeros((nm_nodes,nm_nodes),dtype='float32')
	P = np.eye(nm_nodes, dtype='float32')
	for i in range(0,max_step):
		P = alpha*np.dot(P,adj_matrix) + (1-alpha)*P0
		M = M + P

	return M

def PPMI_matrix(M):

	M = scale_sim_mat(M)
	nm_nodes = len(M)

	col_s = np.sum(M, axis=0).reshape(1,nm_nodes)
	row_s = np.sum(M, axis=1).reshape(nm_nodes,1)
	D = np.sum(col_s)
	rowcol_s = np.dot(row_s,col_s)
	#PPMI = np.log(np.divide(D * M, rowcol_s))
	PPMI = np.log(np.divide(D*M,rowcol_s))
	PPMI[np.isnan(PPMI)] = 0.0
	PPMI[np.isinf(PPMI)] = 0.0
	PPMI[np.isneginf(PPMI)] = 0.0
	PPMI[PPMI<0] = 0.0

	return PPMI


def model(data, output_file, validation_split=0.9):

	#load data from  xx into hidde
	md=open('neorons.txt','r')
	line=md.readline()
	print line
	cnmd = open('layer.txt', 'r')
	l = cnmd.readline()


	cnm=arg_as_list(line)
	hidden_neurons = cnm
	hidden_layers=int(l)
	train_n = int(validation_split * len(data))
	batch_size = 50
	e=50
	m=data.shape[1]



	if hidden_neurons[0] == 0:
		train_data = data
		val_data = data
		input_sh = Input(shape=(data.shape[1],))
		encoded = noise.GaussianNoise(0.2)(input_sh)

	elif hidden_neurons[0] == 1:
		train_data = data[:train_n, :]
		val_data = data[train_n:, :]
		'''train_data = np.expand_dims(train_data, axis=0)
		val_data = np.expand_dims(val_data, axis=0)'''
		#train_data=tf.reshape(train_data,(train_n,data.shape[1],1))
		input_sh = Input(shape=(data.shape[1],))
		input_sh1=Reshape((data.shape[1],1))(input_sh)
		encoded = noise.GaussianNoise(0.2)(input_sh1)

	else:
		train_data = data[:train_n, :]
		val_data = data[train_n:, :]
		input_sh = Input(shape=(data.shape[1],))
		encoded = noise.GaussianNoise(0.2)(input_sh)

	print 'ghhhhhhhhhhhhhhh                  ', encoded.ndim
	print hidden_neurons
	print 'layer    ',hidden_layers
	if hidden_neurons[0] == 0:
		batch_size = len(data)
		encoded = GraphConvolution(
			input_dim=data.shape[1],
			output_dim=hidden_neurons[1],
			support=input_sh,
			act=tf.nn.relu,
		)(encoded)
	elif hidden_neurons[0] == 1:
		encoded = Conv1D(filters=1,kernel_size=178-hidden_neurons[1],activation='relu')(encoded)
		encoded = Flatten()(encoded)
		print 'ghhhhhhhhhhhhhhh                  ',encoded.ndim
	elif hidden_neurons[0] == 2:
		encoded = Dense(hidden_neurons[1], activation='relu')(encoded)
	encoded = noise.GaussianNoise(0.2)(encoded)

	for i in range(2,hidden_layers):
		print i,hidden_neurons[i]
		print 'ghhhhhhhhhhhhhhh dense                 ', encoded.ndim
		encoded = Dense(hidden_neurons[i], activation='relu')(encoded)
		encoded = noise.GaussianNoise(0.2)(encoded)

	decoded = Dense(hidden_neurons[-2], activation='relu')(encoded)

	print hidden_neurons[-2]
	for j in range(hidden_layers-3,0,-1):
		print 'jhjjjjj          ',j, hidden_neurons[j]
		decoded = Dense(hidden_neurons[j], activation='relu')(decoded)
		print data.shape[1]
	decoded = Dense(data.shape[1], activation='sigmoid')(decoded)
	autoencoder = Model(inputs=input_sh, outputs=decoded)#bp to train weights
	autoencoder.compile(optimizer='adadelta', loss='mse')

	#checkpointer = ModelCheckpoint(filepath='bestmodel' + output_file + ".hdf5", verbose=1, save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
	if hidden_neurons[0] == 1:
		train_data1 = data[:train_n, :]
		val_data1 = data[train_n:, :]
		train_generator = DataGenerator(batch_size)
		print 'dfffffffffffffff', train_data.shape
		print 'dfffffffffffffff', train_data1.shape
		train_generator.fit(train_data, train_data)

		val_generator = DataGenerator(batch_size)
		val_generator.fit(val_data, val_data)
	else:


	 train_generator = DataGenerator(batch_size)
	 train_generator.fit(train_data,train_data)

	 val_generator = DataGenerator(batch_size)
	 val_generator.fit(val_data, val_data)

	h=autoencoder.fit_generator(train_generator,
		steps_per_epoch=len(data)/batch_size,
		epochs=e,
		validation_data=val_generator,
		validation_steps=len(data),
		callbacks=[earlystopper],
			)
	#print 'jkjjkjkkjjk   ', h.history
	avge=h.history['val_loss'][e-1]
	print 'avge    ',avge
	enco = Model(inputs=input_sh, outputs=encoded)#just select the first embedding part
	enco.compile(optimizer='adadelta', loss='mse')#configuration ?
	reprsn = enco.predict(data,batch_size=batch_size)
	return reprsn,avge


def process_scripts(args):
	
	filename = args.input
	graph_type = args.graph_type
	Ksteps = args.random_surfing_steps
	alpha = args.random_surfing_rate
	output_file = args.output
	#idden_layers = args.hidden_layers#ga
	#print "hhjjhjjvngbhnnnnnnnnnnnn"
	#hidden_neurons = arg_as_list(args.neurons_hiddenlayer)  # ga
	#hidden_neurons = args.neurons_hiddenlayer#ga



	data_mat, node_idx = read_graph(filename, graph_type)
	data = random_surfing(data_mat, Ksteps, alpha)
	data = PPMI_matrix(data)

 	reprsn,avge = model(data,output_file)
	print 'hjjjj               ',len(reprsn)
	data_reprsn = {'embedding':list(reprsn),'node_id':node_idx}
	#print 'hjjhjjjkjk    ',data_reprsn['avgve']
	df = pd.DataFrame(data_reprsn)
	df.to_pickle(output_file+'.pkl')
	pd.read_pickle(output_file+'.pkl')
	a = open('avge.txt', 'w')
	a.write(str(avge))
	a.close()


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
def main():

  parser = ArgumentParser('DNGR',
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
  #print 'hghjhjppppppppppppppppppppppppjj'

  parser.add_argument('--graph_type', default='undirected',
                      help='Undirected or directed graph as edgelist')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument('--random_surfing_steps', default=10, type=int,
                      help='Number of steps for random surfing')

  parser.add_argument('--random_surfing_rate', default=0.98, type=float,
                      help='alpha random surfing')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

 # parser.add_argument('--hidden_layers', nargs=1, type=int,
                      #help='AutoEnocoder Layers')
  #print 'hghjhjjj'

  #parser.add_argument('--neurons_hiddenlayer',
                      #help='Number of Neurons AE.')
  #print "ghhhjjjjjjjjjjjjjjjjjjjooooooooooooooooooooojjj"

  args = parser.parse_args()
  #print "ghhhjjjjjjjjjjjjjjjjjjjjjj"

  process_scripts(args)

if __name__ == '__main__':
	sys.exit(main())	
	
#subprocess.call('~/Desktop/DNGR-Keras-master/DNGR.py --graph_type '+'undirected'+' --input '+'wine.edgelist'+' --output '+'representation',shell=True)