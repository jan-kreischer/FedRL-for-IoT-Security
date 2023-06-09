#!/bin/bash

while :
do
  cat ./es-sensor/buffer | nc -w 2  192.168.1.160 3000
  sleep 1
done