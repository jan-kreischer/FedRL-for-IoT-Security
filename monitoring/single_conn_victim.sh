#!/bin/bash -u

while echo "hello server"; do
sleep 2
done | nc -q 1 127.0.0.1 5000
