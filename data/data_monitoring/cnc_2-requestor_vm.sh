#!/bin/bash -u

while :
do
nc 192.168.1.53 3000 << END
df -h;
free;
ps aux;
ls /etc;
END
echo "listening"
nc -lv -p 4000 -q 1 > out.txt < /dev/null

sleep 1

done
