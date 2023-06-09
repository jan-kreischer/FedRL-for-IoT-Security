# FedRL for IT-Sec

This work pays large tribute to Timo Schenk's work on RL based MTD deployment that can be found [here](https://github.com/Leitou/rl-based-mtd).  
A full documentation of this work including the theoretical background can be found in the [PDF-report](report.pdf).

## 1. Introduction/ Motivation
Federated Reinforcement Learning for Private and Collaborative Selection of Moving Target Defense Mechanisms for IoT Devices Security.

## Content :label:
Managing de security of IoT devices is non-trivial due to the complexity and heterogeneity of such devices. The differentiation from traditional security issues motivates the need to investigate new security approaches applicable to the IoT computing paradigm. Moving- Target-Defense (MTD) is a novel paradigm that addresses proactive and dynamic cyberat- tacks. The basic philosophy behind MTD is that “perfect security” most likely will never be achievable, making it congruent with the current state of IoT security. In that sense, MTD defends against cyberattacks instead of preventing them using various strategies.
Reinforcement Learning (RL) is a powerful and interesting approach to determine the best MTD technique able to mitigate heterogeneous IoT malware. More in detail, RL is suitable for IoT scenarios where supervised and unsupervised models do not work because the trained model must learn in an online fashion based on the suitability of deployed MTDs. However, there are still many problems in the implementation of RL in practical scenarios. For example, many RL algorithms have the problem of learning time caused by low sample efficiency. Therefore, through information exchange between agents, learning speed can be greatly accelerated. The problem of sharing information is that some IoT tasks and scenarios need to prevent agent information leakage and protect its privacy during the application of RL. Federated Reinforcement Learning (FRL) is a novel approach that can reduce the previ- ous limitations thanks to privacy-preserving and collaborative training of agents. However, the FRL concept is a very incipient, and its feasibility and performance have not been analyzed in IoT malware scenarios. Therefore, the main goal of this project consists of creating and evaluating a framework that uses FRL to select the best MTD mechanism for each malware affecting one or more IoT devices.

## Code structure :book:
- **/data**: contains the code for data monitoring, the training data and the data exploration.

- **/prototypes**: contains prototypes 1 - 3 including the three different sensor environment versions.

- **/src**: dependencies which are share between the different prototypes.

- **/state_anomaly_detection**: investigation of different models for state anomaly detection.


