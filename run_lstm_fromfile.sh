xterm -e "python datagen_fromfile.py"& 
sleep 1;

xterm -e "python Pytorch_Bch_YARP.py"& 
sleep 10;
xterm -e "yarpscope --remote /py/lstm_port/prob:o --min 0.0 --max 1.0"& 
xterm -e "yarp read /read_LSTM_out /py/lstm_port:o"&

