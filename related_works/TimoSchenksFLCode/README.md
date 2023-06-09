# Robust and Privacy-preserving AI in Crowdsensing Platforms

## Content :label:
This repository contains all the code and data required to run the experiments, respectively recreate the figures of the report. 
It is implemented using Python 3 and additional libraries. The DL models are built using PyTorch.

## Setup :hammer:
In order to run the code, Python 3.8+ must be used. Subsequently, all the required libraries (listed in the requirements.txt file) must be installed. 
It is recommended to set up a virtual environment (e. g. using venv) and then install the libraries there. 
Additionally, PyTorch must be installed using the installation instructions provided on their [website](https://pytorch.org/get-started/locally/).

## GPU Support :zap:
GPU Support is available in PyTorch. The code is set up to run on the CPU, as most experiments run within a reasonable time on CPU. 
In order to run them on GPU, the instructions provided by PyTorch need to be followed: 
CUDA must be setup properly and all the variables and models must be put on the GPU, e. g. using the built-in .cuda() command.

## Code structure :book:
* /data/: The raw CSVs of the monitoring. The subdirectories are mapped to the device ids as follows:
  * ras-3-1gb: pi3-1gb
  * ras-4-4gb: pi4-4gb
  * ras-4-black: pi4-2b-bc
  * ras-4-white: pi4-2gb-wc
* /data_exploration/: Runnables to generate the visualizations corresponding to the data exploration from Section 3.3
* /federated_anomaly/: Runnables for the experiments from Section 5.3
* /federated_anomaly_detection/: Runnables for the experiments from Section 6.2
* /federated_binary_classification/: Runnables for the experiments from Section 5.4
* /federated_binary_classification_adversarial/: Runnables for the experiments from Section 6.3
* /standalone/: Runnables for the experiments from Chapter 4
* /various/: configurable experiments used for testing and validation, not included in report.

* aggregation.py: Implements the different Aggregation Functions as well as Threshold Selection for a Federation on top of a list of Participants: Specifically those are:
  * FedAvg
  * Trimmed Mean 1 and 2
  * Coordinate Wise Median
* custom_types.py: Implements the specific ENUM classes for this project: Device Types (RaspberryPi), Behaviors etc.
* data_handler.py: Implements the dataset splitting and scaling for a given definiton of Participants: 
Generates train, validation and test sets for them and can further be used to scale them
* models.py: Implements the PyTorch Sequential models of the MLP as well as the Autoencoder
* participants.py: Implements a Participant, which can then either be an AutoEncoderParticipant (for anomaly detection) or an MLPParticipant (for binary classification).
Further implements adversarial participants as subclasses.
* utils.py: Implements common utility functions for multiple Experiments. Includes reporting utilities as well as definition of (balanced) federation participants

