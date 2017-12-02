# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import h5py
from sklearn.preprocessing import LabelEncoder
from levenshtein_distance import levenshtein_distance
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
numpy.random.seed(7)
dropout = 0.5;
#%%
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
step_epoch = 5;
nb_epoch=100
batchsize = 1
nb_file = 4
nb_file_train = 3
drop = 0.5
opt ='adam'
#scale_data = 1/(2*numpy.pi);
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
#%%
for nb_Cell in [30]:
    print('nb_Cell = %d' % nb_Cell)
    a_trainScoreH1=[]
    a_trainScoreH2=[]
    a_trainScoreH3=[]
    a_testScoreH1=[]
    a_testScoreH2=[]
    a_testScoreH3=[]
	
    a_trainScoreClassify1=[]
    a_testScoreClassify1=[]
		
    a_trainScoreClassify2=[]
    a_testScoreClassify2=[]
	
    a_trainScoreClassify3=[]
    a_testScoreClassify3=[]

    a_trainScoreClassify4=[]
    a_testScoreClassify4=[]
	
    a_trainLeStClassify1=[]
    a_trainLeStClassify2=[]
    a_trainLeStClassify3=[]
    a_trainLeStClassify4=[]	
	
    a_testLeStClassify1=[]
    a_testLeStClassify2=[]
    a_testLeStClassify3=[]
    a_testLeStClassify4=[]		
    #for index_filetest in range(nb_file):
    for index_filetest in [3]:
        #index_filetest = 0;
        print("index_filetest : %d" % index_filetest)
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
        targets = prepare_sequence_target(train_BCH[:,0,:])
        model = LSTM_Bch(8, 30, 2)
        #loss_function = nn.NLLLoss()
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)		

        for nb_ep in range(30):
            nb_epoch = (nb_ep+1)*step_epoch
            print("nb_epoch : %d" % nb_epoch)
            for epoch in range(step_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
                ID_trains = numpy.arange(nb_file_train)
                numpy.random.shuffle(ID_trains)
                print(ID_trains)
                for id_ in ID_trains:
            
                    # Step 1. Remember that Pytorch accumulates gradients.
                    # We need to clear them out before each instance
                    model.zero_grad()
            
                    # Also, we need to clear out the hidden state of the LSTM,
                    # detaching it from its history on the last instance.
                    model.hidden = model.init_hidden()
                    
                    # Step 2. Get our inputs ready for the network, that is, turn them into
                    inputs = prepare_sequence_input(trainX[:,id_,:])
                    targets = prepare_sequence_target(train_BCH[:,id_,:])
                    
                    # Step 3. Run our forward pass.
                    bch_scores = model(inputs)
            
                    # Step 4. Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    loss = loss_function(bch_scores, targets)
                    loss.backward()
                    optimizer.step()
            
            savefile = 'ModelState_Pytorch_singletask_total_inSI_SP_outBCH/classify_LSTM_1out_nbC_'+str(nb_Cell)+'_nbE_'+str(nb_epoch)+'_nbTf_'+str(nb_file_train)+'_iTestF_'+str(index_filetest)+'_'+opt+'.pt';
            #torch.save(model,savefile)
            #model = torch.load('filename.pt')
            
            torch.save(model.state_dict(), savefile)
            # See what the scores are after training
            inputs = prepare_sequence_input(testX[:,0,:])
            targets = prepare_sequence_target(test_BCH[:,0,:])
            #print(targets)
            bch_scores = model(inputs)
            #print(bch_scores)           
            prob = F.softmax(bch_scores)
            #print(prob)
            
            index_testSI = numpy.argmax(test_SI, axis=2)
            index_testSP = numpy.argmax(test_SP, axis=2)
            index_testIU = numpy.argmax(test_IU, axis=2)
            index_testFX = numpy.argmax(test_FX, axis=2)
            index_testBCH = numpy.argmax(test_BCH, axis=2)
            testFX = numpy.argmax(test_FX, axis=2)
            testBCH = numpy.argmax(test_BCH, axis=2)	
            savefile_test = 'Data_Pytorch_singletask_total_inSI012_4sbj_noBC_SP_outBCH/test_predict_classify_LSTM_1out_nbC_'+str(nb_Cell)+'_nbE_'+str(nb_epoch)+'_nbTf_'+str(nb_file_train)+'_iTestF_'+str(index_filetest)+'_'+opt+'.data';
            with open(savefile_test, "w") as f:
                for i in range(min_len):
                    f.write("--SI:%d --SP:%d --IU:%d --FX:%d  --BCH_real:%d --BCH_pred0:%f --BCH_pred1:%f\n" %(index_testSI[i,0],index_testSP[i,0],index_testIU[i,0],index_testFX[i,0],index_testBCH[i,0],prob.data[i,0],prob.data[i,1]))		

	

