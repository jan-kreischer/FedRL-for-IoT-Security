#!/bin/bash -u

# system dependencies
apt-get update
apt install libatlas3-base -y # for numpy
apt-get install gfortran libatlas-base-dev libopenblas-dev liblapack-dev -y # for scikit-learn
apt install python3-pip -y
apt-get install net-tools -y
apt-get install arp-scan -y
apt-get install git -y

# python dependencies
python3 -m pip install --upgrade pip
python3 -m pip install scipy==1.3.3 # last version for which piwheel is available (speed)
python3 -m pip install -U scikit-learn # installs all other dependencies automatically
python3 -m pip install setproctitle
python3 -m pip install jsonschema
python3 -m pip install pandas
python3 -m pip install pycryptodome # for ransomware poc

# ensure monitoring permissions
chmod +x rl_sampler_online.sh
