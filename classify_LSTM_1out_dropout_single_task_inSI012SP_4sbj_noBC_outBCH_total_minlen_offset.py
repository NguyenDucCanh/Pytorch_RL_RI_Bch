# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Bidirectional
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import h5py
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from levenshtein_distance import levenshtein_distance

for nb_Cell in [30]:
	
	step_epoch = 5;
	print('nb_Cell = %d' % nb_Cell)
	nb_epoch=100
	batchsize = 1
	nb_file = 4
	nb_file_train = 3
	drop = 0.5
	opt ='adam'
	#nb_Cell =20
	#nb_epoch=20
	#batchsize = 1
	#nb_file = 3z
	#nb_file_train = 2
	#drop = 0.8
	
	scale_data = 1/(2*numpy.pi);
	# fix random seed for reproducibility
	numpy.random.seed(7)
	# load the dataset
	#dataframe = pandas.read_csv('data.txt', usecols=[1], engine='python')
	
	
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
		data_head = data_head*scale_data;
		data_head = data_head.astype('float32')
		data_head = data_head[0:min_len,:]
		
		len_data = min_len
		
		IU = to_categorical(IU,maxVal_IU)
		SI = to_categorical(SI,maxVal_SI)
		SP = to_categorical(SP,maxVal_SP)
		FX = to_categorical(FX,maxVal_FX)
		BCH = to_categorical(BCH,maxVal_BCH)
		
		IU = numpy.reshape(IU, (1,len_data, maxVal_IU))
		SI = numpy.reshape(SI, (1,len_data, maxVal_SI))
		SP = numpy.reshape(SP, (1,len_data, maxVal_SP))
		FX = numpy.reshape(FX, (1,len_data, maxVal_FX))
		BCH = numpy.reshape(BCH, (1,len_data, maxVal_BCH))
				
		data_head = numpy.reshape(data_head,(1,len_data,3))
		
		return (IU,SI,SP,FX,BCH,data_head)
	
	(IU,SI,SP,FX,BCH,data_head)=load_file(0)
	
	for i in range(nb_file-1):
		(IU_k,SI_k,SP_k,FX_k,BCH_k,data_head_k)=load_file(i+1)
		IU = numpy.concatenate((IU,IU_k), axis=0)
		SI = numpy.concatenate((SI,SI_k), axis=0)
		SP = numpy.concatenate((SP,SP_k), axis=0)
		FX = numpy.concatenate((FX,FX_k), axis=0)
		BCH = numpy.concatenate((BCH,BCH_k), axis=0)
		data_head = numpy.concatenate((data_head,data_head_k),axis=0)
	
	
	def classification_evaluation(Predict,Target):
		classification_score = 0
		for i in range(len(Predict)):
			if int(Predict[i]) == int(Target[i]):
				classification_score = classification_score+1
		classification_score = classification_score/float(len(Predict))
		return classification_score
	
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
	
	for index_filetest in range(nb_file):
		train_IU = numpy.delete(IU,(index_filetest),axis=0)
		train_SI = numpy.delete(SI,(index_filetest),axis=0)
		train_SP = numpy.delete(SP,(index_filetest),axis=0)
		#train_GT = numpy.delete(GT,(index_filetest),axis=0)
		train_FX = numpy.delete(FX,(index_filetest),axis=0)
		train_head = numpy.delete(data_head,(index_filetest),axis=0)
		train_BCH = numpy.delete(BCH,(index_filetest),axis=0)
		
		test_IU = IU[index_filetest]
	
		#train_SI = SI[index_filetest]
		#train_SP = SP[index_filetest]
		#train_head = data_head[index_filetest]
		#train_SI = numpy.reshape(train_SI, (1,min_len, maxVal_SI))
		#train_SP = numpy.reshape(train_SP, (1,min_len, maxVal_SP))
		#train_head = numpy.reshape(train_head,(1,min_len,3))
			
		test_SI = SI[index_filetest]
		test_SP = SP[index_filetest]
		#test_GT = GT[index_filetest]
		test_FX = FX[index_filetest]
		test_BCH = BCH[index_filetest]
		test_head = data_head[index_filetest]
		
		test_IU = numpy.reshape(test_IU, (1,min_len, maxVal_IU))
		test_SI = numpy.reshape(test_SI, (1,min_len, maxVal_SI))
		test_SP = numpy.reshape(test_SP, (1,min_len, maxVal_SP))
		#test_GT = numpy.reshape(test_GT, (1,min_len, maxVal_GT))
		test_FX = numpy.reshape(test_FX, (1,min_len, maxVal_FX))
		test_BCH = numpy.reshape(test_BCH, (1,min_len, maxVal_BCH))
		test_head = numpy.reshape(test_head, (1,min_len, 3))
	
		trainX = numpy.concatenate((train_SI,train_SP),axis=2)
		testX = numpy.concatenate((test_SI,test_SP),axis=2)
		
		print('Cross index file = %d' % index_filetest)
		input_shape = maxVal_SI+maxVal_SP
		inputs = Input(shape=(None,input_shape),name = 'inputs')
		lstm_out = LSTM(nb_Cell,return_sequences=True,)(inputs)
		drop_out = Dropout(drop)(lstm_out)
		output4 = Dense(maxVal_BCH,activation="softmax",name='output4')(drop_out)
		model = Model(input=inputs, output = output4)
		#model.coSIile(optimizer='rmsprop', loss = {'output1':'categorical_crossentropy','output2':'categorical_crossentropy','output3':'categorical_crossentropy'},loss_weights={'output1': 1., 'output2': 1., 'output3': 1.})
		#model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy')
		model.compile(optimizer=opt, loss = 'categorical_crossentropy')
		
		for nb_ep in range(30):
			nb_epoch = (nb_ep+1)*step_epoch;
			model.fit(trainX,train_BCH, nb_epoch=step_epoch,batch_size=1, verbose=2)
			
			#savefile = 'Model_dropout_singletask_total_inSI_SP_outBCH/classify_LSTM_1out_nbC_'+str(nb_Cell)+'_nbE_'+str(nb_epoch)+'_nbTf_'+str(nb_file_train)+'_iTestF_'+str(index_filetest)+'_'+opt+'.h5';
			#model.save(savefile) 
			
			trainPredict4 = model.predict(trainX)	
			testPredict4 = model.predict(testX)
			
			#index_testPredict1 = numpy.argmax(testPredict1, axis=2)
			#index_testPredict2 = numpy.argmax(testPredict2, axis=2)
			#index_testPredict3 = numpy.argmax(testPredict3, axis=2)
			index_testPredict4 = numpy.argmax(testPredict4, axis=2)
			
			#index_testTarget1 = numpy.argmax(test_IU, axis=2)
			#index_testTarget2 = numpy.argmax(test_GT, axis=2)
			#index_testTarget3 = numpy.argmax(test_FX, axis=2)
			index_testTarget4 = numpy.argmax(test_BCH, axis=2)
			
			#index_trainPredict1 = numpy.argmax(trainPredict1, axis=2)
			#index_trainPredict2 = numpy.argmax(trainPredict2, axis=2)
			#index_trainPredict3 = numpy.argmax(trainPredict3, axis=2)
			index_trainPredict4 = numpy.argmax(trainPredict4, axis=2)
			
			#index_trainTarget1 = numpy.argmax(train_IU, axis=2)
			#index_trainTarget2 = numpy.argmax(train_GT, axis=2)
			#index_trainTarget3 = numpy.argmax(train_FX, axis=2)
			index_trainTarget4 = numpy.argmax(train_BCH, axis=2)
		
			index_testSI = numpy.argmax(test_SI, axis=2)
			index_testSP = numpy.argmax(test_SP, axis=2)
			index_testIU = numpy.argmax(test_IU, axis=2)
			index_testFX = numpy.argmax(test_FX, axis=2)
			index_testBCH = numpy.argmax(test_BCH, axis=2)	
			savefile_test = 'Data_dropout_singletask_total_inSI012_4sbj_noBC_SP_outBCH/test_predict_classify_LSTM_1out_nbC_'+str(nb_Cell)+'_nbE_'+str(nb_epoch)+'_nbTf_'+str(nb_file_train)+'_iTestF_'+str(index_filetest)+'_'+opt+'.data';
			with open(savefile_test, "w") as f:
				for i in range(len(testPredict4[0])):
					f.write("--SI:%d --SP:%d --IU:%d --FX:%d  --BCH_real:%d --BCH_pred0:%f --BCH_pred1:%f\n" %(index_testSI[0,i],index_testSP[0,i],index_testIU[0,i],index_testFX[0,i],index_testBCH[0,i],testPredict4[0,i,0],testPredict4[0,i,1],))		
	

