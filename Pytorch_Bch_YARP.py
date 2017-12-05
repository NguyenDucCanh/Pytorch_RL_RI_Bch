#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:03:36 2017

@author: canh
"""
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
import  threading
torch.manual_seed(1)
numpy.random.seed(7)
dropout = 0.5;
#%% Define LSTM_model
class LSTM_Bch(nn.Module):
    def __init__(self, input_dim, hidden_dim,bch_dim):
        super(LSTM_Bch, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim,dropout = dropout)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2bch = nn.Linear(hidden_dim, bch_dim)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(
            sequence.view(len(sequence), 1, -1), self.hidden)
        bch_out = self.hidden2bch(lstm_out.view(len(sequence), -1))
        #tag_scores = F.log_softmax(tag_space)
        #tag_scores = tag_space
        return bch_out

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
#%% Load Model
index_filetest = 3
savefile = 'ModelState_Pytorch_singletask_total_inSI_SP_outBCH/classify_LSTM_1out_nbC_'+str(nb_Cell)+'_nbE_'+str(nb_epoch)+'_nbTf_'+str(nb_file_train)+'_iTestF_'+str(index_filetest)+'_'+opt+'.pt';
model = LSTM_Bch(8, 30, 2)
#torch.save(model.state_dict(), savefile)
model.load_state_dict(torch.load(savefile))
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

inputs = prepare_sequence_input(trainX[:,0,:])
bch_scores = model(inputs)
print(bch_scores.data[1:10,0])
prob = F.softmax(bch_scores)
"""
# Test input one sequence is the same with the one-by-one input or not
#inputs = prepare_sequence_input(trainX[:,0,:])
#bch_scores = model(inputs)
#print(bch_scores.data[1:10,0])

##======================================
#result_sequence = []
#for i in range(len(trainX)):
#    inputs = prepare_one_input(trainX[i,0,:])
#    bch_scores = model(inputs)
#    result_sequence.append(bch_scores.data[0,0])
#print(result_sequence[1:10])
"""

#%%
import optparse
import yarp
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option('--thres', dest='threshold', default=0.23, type='float',
                      help='[Threshold trigger BCH default = 0.23 for read from file], for real-time: 0.035')

(options, args) = parser.parse_args()
yarp.Network.init()
p_BCH = yarp.BufferedPortBottle()
p_BCH.open("/py/lstm_port:o");

p_BCH_prob = yarp.BufferedPortBottle()
p_BCH_prob.open("/py/lstm_port/prob:o");

#yarp.Network.connect("/keras_FX","/receive_gaze_arm_process")
#yarp.Network.connect("/keras_GT","/receive_gaze_arm_process")

bottle_BCH = p_BCH.prepare()
#bottle_GT = p_GT.prepare()

current_BCH = -1;
#%%
#from threading import thread, Lock
import threading, time
data_in = numpy.zeros([8]);
default_port_in = '/py/lstm_port:i'
class DataPort(yarp.BufferedPortBottle):
    def onRead(self, bot, *args, **kwargs):
        print 'Bottle [%s], args [%s], kwargs [%s]' % (bot.toString(), args, kwargs)

class DataProcessor(yarp.BottleCallback):
    def onRead(self, bot, *args, **kwargs):
        #my_mutex = threading.Lock() # CAnnot use MUTEX now
        for i in range(bot.size()):
            data_in[i] = bot.get(i).asDouble()
            #print bot.size()
            #print bot.get(i).asDouble()
        #my_mutex.release()
        #print 'Bottle [%s]' % (bot.toString())
        
port = DataPort()
proc = DataProcessor()
port.useCallback(proc)
port.open(default_port_in)
yarp.Network.connect('/py/datagen:o', default_port_in)
threshold = options.threshold
#%%
"""
def foo():
    global data_in
    global current_BCH
    global predict_BCH
    while 1:
        #print data_in
        inputs = prepare_one_input(data_in)
        #print inputs
        bch_scores = model(inputs)
        prob = F.softmax(bch_scores)
        if (prob.data[0,0] > threshold):
            predict_BCH = 1
        else:
            predict_BCH = 0;
        #print ('i = %d, BCH_real: %d, BCH_pred:%d\n' % (i,1-numpy.argmax(test_BCH[i,0,:]),predict_BCH))
        if current_BCH != predict_BCH:
            bottle_BCH = p_BCH.prepare()
            bottle_BCH.clear()
            bottle_BCH.addString("bch")
            bottle_BCH.addInt(predict_BCH)
            p_BCH.writeStrict()
            current_BCH = predict_BCH
            print "out bch [%d]" % (predict_BCH)
        yarp.Time.delay(0.04) # it can solver the problem of incremental memory due to exponantial threading call, but can be delay due to calculating time
    #threading.Timer(0.04, foo).start()

foo();
"""
#%% Simple way to avoid threading
import time
starttime=time.time()
period = 0.04;
while True:
    #print "tick"
    #print data_in
    inputs = prepare_one_input(data_in)
    #print inputs
    bch_scores = model(inputs)
    prob = F.softmax(bch_scores)
    bottle_BCH_prob = p_BCH_prob.prepare()
    bottle_BCH_prob.clear()
    bottle_BCH_prob.addDouble(prob.data[0,0])
    p_BCH_prob.writeStrict()
    
    if (prob.data[0,0] > threshold):
        predict_BCH = 1
    else:
        predict_BCH = 0;
    #print ('i = %d, BCH_real: %d, BCH_pred:%d\n' % (i,1-numpy.argmax(test_BCH[i,0,:]),predict_BCH))
    if current_BCH != predict_BCH:
        bottle_BCH = p_BCH.prepare()
        bottle_BCH.clear()
        bottle_BCH.addString("bch")
        bottle_BCH.addInt(predict_BCH)
        p_BCH.writeStrict()
        current_BCH = predict_BCH
        print "out bch [%d]" % (predict_BCH)
    time.sleep(period - ((time.time() - starttime) % period))

#%% Close port
port.close()
port.interrupt()
p_BCH.close()
p_BCH.interrupt()

p_BCH_prob.close()
p_BCH_prob.interrupt()
#%%
"""
i = 0;
len_seq = testX.shape[0]

#result_sequence = []
#for i in range(len(trainX)):
#    inputs = prepare_one_input(testX[i,0,:])
#    bch_scores = model(inputs)
#    result_sequence.append(bch_scores.data[0,0])
#print(result_sequence[1:10])
#prob = F.softmax(bch_scores)
threshold = 0.25
def foo():
    global i
    global len_seq
    global current_BCH
    global game
    global predict_BCH
	#print('i: %d, len_IU: %d, current_FX: %d\n' %(i,len_IU,current_FX))
    if i < len_seq-1:
        inputs = prepare_one_input(testX[i,0,:])
        bch_scores = model(inputs)
        prob = F.softmax(bch_scores)
        if (prob.data[0,0] > threshold):
            predict_BCH = 1
        else:
            predict_BCH = 0;
        print ('i = %d, BCH_real: %d, BCH_pred:%d\n' % (i,1-numpy.argmax(test_BCH[i,0,:]),predict_BCH))
        if current_BCH != predict_BCH:
            bottle_BCH = p_BCH.prepare()
            bottle_BCH.clear()
            bottle_BCH.addString("bch")
            bottle_BCH.addInt(predict_BCH)
            p_BCH.writeStrict()
            current_BCH = predict_BCH
        i = i+1;
        threading.Timer(0.04, foo).start()
foo();
print game;
"""