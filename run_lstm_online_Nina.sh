#xterm -e "python datagen.py"& 
#sleep 1;

xterm -e "python Pytorch_Bch_YARP.py --thres 0.03"& 
sleep 10;
xterm -e "SOMBRERO_process_event_to_python"& # src:/home/canh/WORKING/Autonomous_Nina/Gestures_process/process_event_to_python
sleep 5;
xterm -e "yarp read /read_SI_SS /process_event_to_python:o"&
xterm -e "yarp read /read_LSTM_out /py/lstm_port:o"&
xterm -e "yarpscope --remote /py/lstm_port/prob:o --min 0.0 --max 0.1"& 
#xterm -e "yarp write /write_SI_SS"& 
#yarp connect /write_SI_SS /process_event_to_python:i&
yarp connect /process_event_to_python:o /py/lstm_port:i&


