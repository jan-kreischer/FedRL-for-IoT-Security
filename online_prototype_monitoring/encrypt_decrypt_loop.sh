#!/bin/bash -u

# used for decision state monitoring and afterstates in case of incorrect MTDs

attackpath="/data/attacks/Ransomware-PoC/main.py"
dir2encrypt="/data/online_prototype_monitoring/Ransomware/TestFileStructure"
while :
do
  # encrypt directory TestFileStructure
  echo "encrypting"
  python "$attackpath" -p "$dir2encrypt" -e


  # decrypt directory
  echo "decrypting"
  python "$attackpath" -p "$dir2encrypt" -d
done
