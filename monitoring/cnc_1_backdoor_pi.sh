#!/bin/bash -u

while :
do
cat ./ranfile.txt | nc -w 2 192.168.1.50 3000
sleep 2
done

# could also send ./es-sensor/buffer, data specific to electrosense