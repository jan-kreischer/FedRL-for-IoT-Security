# Optimizing MTD Deployment on IoT Devices using Reinforcement Learning

## Content :label:
This repository contains all the code and data required to run the RL-based, reactive MTD selection experiments, 
respectively recreate the figures of the report. It is implemented using Python 3 and additional libraries. 
TheRL/DL models are built using PyTorch. Custom Environment interfaces work analogously to OpenAI Gym.


## Code structure :book:
- datasets
- data exploration to check data availability, timeline and distribution of features
- shell scripts required for monitoring, incl. data leak poc attacks
- 3 offline/simulated environments to train/pretrain an agent
- 1 controller for full online, on-device learning
- 1 mock controller, excluding the ML parts that is used to monitor realistic data for pretraining
- anomaly detection tests