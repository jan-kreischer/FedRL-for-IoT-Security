#!/bin/bash -u

while :
do
# Sends the content of the ranfile via the netcat command over TCP/UDP to the specified IP Address 192.168.1.50 on port 3000.
cat ./ranfile.txt | nc -w 2 192.168.1.50 3000
sleep 2
done

# could also send ./es-sensor/buffer, data specific to electrosense