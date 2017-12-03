#xterm -e "python datagen.py"& 
#sleep 1;

xterm -e "python Pytorch_Bch_YARP.py"& 
sleep 10;
xterm -e "SOMBRERO_process_event_to_python"& # src:/home/canh/WORKING/Autonomous_Nina/Gestures_process/process_event_to_python
sleep 5;
xterm -e "yarp read /read_SI_SS /process_event_to_python:o"&
xterm -e "yarp read /read_LSTM_out /py/lstm_port:o"&
#xterm -e "yarp write /write_SI_SS"& 
#yarp connect /write_SI_SS /process_event_to_python:i&
yarp connect /process_event_to_python:o /py/lstm_port:i
sleep 1;
# comment in "yarp write port 
# SI-1:speaking,2:questioning, 3:listening
# SS-1:listening, 2:correct answer, 3:speaking
echo  "SI 1"| yarp write /write_SI_SS /process_event_to_python:i # SI-introduction
sleep 3;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 0.5;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-correct answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 1;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-correct answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  "SI 1"| yarp write /write_SI_SS /process_event_to_python:i # SI-introduction
sleep 3;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 0.5;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-wrong answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 1;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-correct answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  "SI 1"| yarp write /write_SI_SS /process_event_to_python:i # SI-introduction
sleep 3;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 0.5;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-wrong answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 1;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-correct answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  "SI 1"| yarp write /write_SI_SS /process_event_to_python:i # SI-introduction
sleep 3;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 0.5;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-wrong answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 1;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-correct answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  "SI 1"| yarp write /write_SI_SS /process_event_to_python:i # SI-introduction
sleep 3;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 0.5;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-wrong answer
sleep 0.5;
echo  SS 1|yarp write /write_SI_SS /process_event_to_python:i # SS-listening
sleep 1;
echo  SI 2|yarp write /write_SI_SS /process_event_to_python:i # SI-questioning
sleep 3;
echo  SI 3|yarp write /write_SI_SS /process_event_to_python:i # SI-listening
sleep 1;
echo  SS 2|yarp write /write_SI_SS /process_event_to_python:i # SS-correct answer
sleep 0.5;

