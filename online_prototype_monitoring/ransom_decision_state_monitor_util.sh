#!/bin/bash -u


while :
do
  # encrypt directory TestFileStructure
  python main.py -p "/data/TestFileStructure" -e
  # decrypt directory
  python main.py -p "/data/TestFileStructure" -d
done



