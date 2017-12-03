xterm -e "python datagen_fromfile.py"& 
sleep 1;

xterm -e "python Pytorch_Bch_YARP.py"& 
sleep 3;
xterm -e "yarp read /read_LSTM_out /py/lstm_port:o"&

