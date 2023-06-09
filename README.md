# FedRL for IT-Sec

This work pays large tribute to Timo Schenk's work on RL based MTD deployment that can be found [here](https://github.com/Leitou/rl-based-mtd).

Optimizing MTD Deployment on IoT Devices using Reinforcement Learning

## Content :label:
This repository contains all the code and data required to run the RL-based, reactive MTD selection experiments, 
respectively recreate the figures of the report. It is implemented using Python 3 and additional libraries. 
TheRL/DL models are built using PyTorch. Custom Environment interfaces work analogously to OpenAI Gym.
All has been tested on a Raspberry Pi 3 Model B+ with 1 GB RAM.


## Code structure :book:
- data/: datasets for 1. raw behaviors, and 2. for 'real' environment states (decision-/afterstate data) as perceived by an online agent
- data_exploration/: code to check data availability, timeline and distribution of features, PCA dimensionality reduction tests
- monitoring/: shell scripts required for monitoring/observing the datasets, i.e. the environment, incl. data leak poc attacks
- offline_prototype_1_raw_behaviors/: simulation of learning progress to explore problem domain of reactive RL-based MTD under ideal conditions (raw behavior data + supervised rewards)
- offline_prototype_2_raw_behaviors/: simulation of learning progress of a completely unsupervised learning setup and the impact of using anomaly detection (raw behavior data + unsupervised rewards via autoencoder)
- offline_prototype_3_ds_as_sampling/: simulation of learning progress to pretrain an agent offline as realistically as possible to allow policy transfer to a full online setup (decision- and afterstate data as observed by a full online agent + unsupervised rewards via autoencoder)
- online_prototype_1_ondevice/: fully operational online agent, including controller + components for complete online, on-device learning (incl. setup.sh for dependency installation)
- online_prototype_monitoring/: mock agent for testing purposes with ML-parts as comments that is used to monitor realistic data for pretraining (incl. setup.sh for dependency installation)
- resource_evaluation/: an evaluation of the resource consumption of the full online agent in a worstcase scenario (ransomware_poc attack + mitigation via directory trap MTD) monitored via nmon on a Raspberry Pi 3 Model B+, 1GB RAM
- utils/: helper functions to train and evaluate autoencoders and agents
- agent.py: DQ-Learning based agent code
- anomaly_detection_tests.py: experiments with different anomaly detectors than the autoencoder used in the prototypes (Local Outlier Factor, Isolation Forest and One Class SVM)
- autoencoder.py: model, training, threshold selection and anomaly detection
- data_provider.py: code to read, preprocess (clean + scale, evtl pca transform) and provide data for simulated environments
- simulation_engine.py: replay memory initialization and agent-environment interaction loop for offline prototypes.
