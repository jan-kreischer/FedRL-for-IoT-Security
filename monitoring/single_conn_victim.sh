#!/bin/bash -u

# Send "hello server" every 2 seconds to the specified ip address
while echo "hello server"; do
sleep 2
done | nc -q 1 127.0.0.1 5000
