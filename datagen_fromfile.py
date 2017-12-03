#!/usr/bin/env python
#
#Author Duc-Canh NGUYEN

#!/usr/bin/python

import yarp
import optparse
import time
import math
import random

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime, threading
torch.manual_seed(1)
numpy.random.seed(7)
dropout = 0.5;
#%%
nb_Cell = 30
step_epoch = 5;
nb_epoch=115
batchsize = 1
nb_file = 4
nb_file_train = 3
drop = 0.5
opt ='adam'

maxVal_IU,maxVal_SI,maxVal_SP,maxVal_FX,maxVal_BCH = 3,5,3,4,2
def to_categorical(seq_data,maxVal):
    category_data = numpy.zeros((len(seq_data),maxVal))
    for i in range(len(seq_data)):
        if seq_data[i] >= maxVal:
            category_data[i,maxVal-1] = 1.0
        else: 
            category_data[i,seq_data[i]-1]=1.0
    return category_data
def find_min(n):
    min_len = 100000000
    for i in range(n):
        filename = '_full_new_clap_new_SI_noBC/data_'+str(i+1)+'.txt';
        data_in = pandas.read_csv(filename, usecols=[0], engine='python',header=None, sep=r"\s+")
        len_data = len(data_in)
        if min_len > len_data:
            min_len = len_data
    return min_len
min_len = find_min(nb_file)
def prepare_sequence_input(seq): # convert sequence to tensor pytorch
    seq = numpy.reshape(seq,(min_len,1,-1))
    tensor = torch.from_numpy(seq)
    tensor = tensor.type(torch.FloatTensor) # This is MUST be Float Tensor
    return autograd.Variable(tensor)

def prepare_one_input(seq): # convert sequence to tensor pytorch
    seq = numpy.reshape(seq,(1,1,-1))
    tensor = torch.from_numpy(seq)
    tensor = tensor.type(torch.FloatTensor) # This is MUST be Float Tensor
    return autograd.Variable(tensor)

def prepare_sequence_target(seq): # convert sequence to tensor pytorch
    seq = numpy.reshape(seq,(min_len,1,-1))
    seq = numpy.argmax(seq, axis=2)
    seq = numpy.reshape(seq,(min_len)) #MUST BE SCALAR
    tensor = torch.from_numpy(seq)
    tensor = tensor.type(torch.LongTensor) # This is MUST be Long Tensor?
    return autograd.Variable(tensor)

def load_file(i):
    filename = '_full_new_clap_new_SI_noBC/data_'+str(i+1)+'.txt';
    data = pandas.read_csv(filename, engine='python',header=None, sep=r"\s+")
    data = data.values
    data = data.astype('int')
    IU = data[0:min_len,0]
    SI = data[0:min_len,1]
    SP = data[0:min_len,2]
    FX = data[0:min_len,3]
    BCH = data[0:min_len,4]
    filename_head = '_full_new_clap_new_SI_noBC/head_'+str(i+1)+'.txt';
    data_head = pandas.read_csv(filename_head, engine='python',header=None, sep=r"\s+")	
    data_head = data_head.values
    data_head = data_head;
    data_head = data_head.astype('float32')
    data_head = data_head[0:min_len,:]
    len_data = min_len
    IU = to_categorical(IU,maxVal_IU)
    SI = to_categorical(SI,maxVal_SI)
    SP = to_categorical(SP,maxVal_SP)
    FX = to_categorical(FX,maxVal_FX)
    BCH = to_categorical(BCH,maxVal_BCH)
    IU = numpy.reshape(IU, (len_data,1, maxVal_IU))
    SI = numpy.reshape(SI, (len_data,1, maxVal_SI))
    SP = numpy.reshape(SP, (len_data,1, maxVal_SP))
    FX = numpy.reshape(FX, (len_data,1, maxVal_FX))
    BCH = numpy.reshape(BCH, (len_data,1, maxVal_BCH))		
    data_head = numpy.reshape(data_head,(len_data,1,3))
    return (IU,SI,SP,FX,BCH,data_head)

(IU,SI,SP,FX,BCH,data_head)=load_file(0)

def to_categorical(seq_data,maxVal):
    category_data = numpy.zeros((len(seq_data),maxVal))
    for i in range(len(seq_data)):
        if seq_data[i] >= maxVal:
            category_data[i,maxVal-1] = 1.0
        else: 
            category_data[i,seq_data[i]-1]=1.0
    return category_data

for i in range(nb_file-1):
    (IU_k,SI_k,SP_k,FX_k,BCH_k,data_head_k)=load_file(i+1)
    IU = numpy.concatenate((IU,IU_k), axis=1)
    SI = numpy.concatenate((SI,SI_k), axis=1)
    SP = numpy.concatenate((SP,SP_k), axis=1)
    FX = numpy.concatenate((FX,FX_k), axis=1)
    BCH = numpy.concatenate((BCH,BCH_k), axis=1)
    data_head = numpy.concatenate((data_head,data_head_k),axis=1)


def classification_evaluation(Predict,Target):
    classification_score = 0
    for i in range(len(Predict)):
        if int(Predict[i]) == int(Target[i]):
            classification_score = classification_score+1
    classification_score = classification_score/float(len(Predict))
    return classification_score
#%% Load Data
    
index_filetest = 3
train_IU = numpy.delete(IU,(index_filetest),axis=1)
train_SI = numpy.delete(SI,(index_filetest),axis=1)
train_SP = numpy.delete(SP,(index_filetest),axis=1)
#GT = numpy.delete(GT,(index_filetest),axis=0)
train_FX = numpy.delete(FX,(index_filetest),axis=1)
train_head = numpy.delete(data_head,(index_filetest),axis=1)
train_BCH = numpy.delete(BCH,(index_filetest),axis=1)
test_IU = IU[:,index_filetest,:]
test_SI = SI[:,index_filetest,:]
test_SP = SP[:,index_filetest,:]
#test_GT = GT[index_filetest]
test_FX = FX[:,index_filetest,:]
test_BCH = BCH[:,index_filetest,:]
test_head = data_head[:,index_filetest,:]
test_IU = numpy.reshape(test_IU, (min_len,1, maxVal_IU))
test_SI = numpy.reshape(test_SI, (min_len,1, maxVal_SI))
test_SP = numpy.reshape(test_SP, (min_len,1, maxVal_SP))
#test_GT = numpy.reshape(test_GT, (min_len,1, maxVal_GT))
test_FX = numpy.reshape(test_FX, (min_len,1, maxVal_FX))
test_BCH = numpy.reshape(test_BCH, (min_len,1, maxVal_BCH))
test_head = numpy.reshape(test_head, (min_len,1, 3))
trainX = numpy.concatenate((train_SI,train_SP),axis=2)
testX = numpy.concatenate((test_SI,test_SP),axis=2)
#%%
period = 0.04 # sec
yarp.Network.init()
default_port = '/py/datagen:o'

port = yarp.BufferedPortBottle()
port.open(default_port)
inittime = time.time()
for i in xrange(len(testX)):
    bottle = port.prepare()
    bottle.clear()
    for j in xrange(len(testX[i,0,:])):
        #botlist = bottle.addList()
        bottle.addDouble(testX[i,0,j])
    time.time()
    print "[%d,%.3f] %s" % (i+1, time.time() - inittime, bottle.toString())
    port.write()
    yarp.Time.delay(period)
port.close()
port.interrupt()
yarp.Network.fini()
#%%
"""
yarp.Network.init()
def main():
    print 'YARP Random Data Generator from file'

    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    parser.add_option('-n', '--number', dest='number', default=1000, type="int", help='Number of samples to send')
    parser.add_option('-p', '--port', dest='port', default='/py/datagen:o', help='name of the output port')
    parser.add_option('-l', '--lists', dest='lists', default=1, type='int', help='number of lists in bottle')
    parser.add_option('-c', '--components', dest='components', default=3, type='int', help='number of components in each list')
    parser.add_option('-f', '--frequency', dest='frequency', default=10., type='float', help='sampling frequency (Hz)')
    parser.add_option('-m', '--mean', dest='mean', default=0., type='float', help='mean for the gaussian PRNG')
    parser.add_option('-s', '--standard-deviation', dest='stddev', default=1., type='float', help='standard deviation for the gaussian PRNG')
    parser.add_option('--sin', dest='sin', default=None, type='float', help='generate sin function')
    parser.add_option('--seed', dest='seed', default=None, type='int', help='seed for the PRNG')
    (options, args) = parser.parse_args()

    port = yarp.BufferedPortBottle()
    port.open(options.port)
    
    random.seed(options.seed)

    print 'Port:', options.port
   
    inittime = time.time()    
    for i in xrange(options.number):
        bottle = port.prepare()
        bottle.clear()
        for j in xrange(options.lists):
            botlist = bottle.addList()
            for k in xrange(options.components):
                if not options.sin:
                    botlist.addDouble(random.gauss(options.mean, options.stddev))
                else:
                    botlist.addDouble(math.sin(time.time() * options.sin))
            
        print "[%d,%.3f] %s" % (i+1, time.time() - inittime, bottle.toString())
        port.write()
        yarp.Time.delay(1. / options.frequency)
    
    port.close()
    port.interrupt()
    
    yarp.Network.fini()


if __name__ == "__main__":
    main()
"""