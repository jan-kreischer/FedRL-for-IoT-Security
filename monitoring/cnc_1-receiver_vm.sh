#!/bin/bash

while :
do
# -l specifies that nc should listen for an incoming connection
# -v verbose
# This script is used to listen and wait for replies from the slave devices.
  nc -lv 4000;
done
