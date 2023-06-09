from typing import Dict
from collections import defaultdict
from custom_types import MTDTechnique, Behavior

# TODO: main script for prototype one
#  define MTDs & attacks mapping
#  start a for loop for number of episodes
#  generate an episode by sampling from data and using MTD-Attack mapping to decide when an episode terminates
#  (use a pretrained autoencoder to predict normal/malicious after deployment)
#  choose a DRL method/neural network that gets updated in every step/at the end of episodes, while last step before end is valued most?.
#  train


# define MTD - Attack Mapping
# TODO: Multiple attacks to same MTD, same attack to multiple MTD, i.e. Ransomware?
supervisor_map: Dict[Behavior, MTDTechnique] = defaultdict(lambda: MTDTechnique.NO_MTD, {
    Behavior.NORMAL: MTDTechnique.NO_MTD,
    Behavior.CNC_BACKDOOR_JAKORITAR: MTDTechnique.CNC_IP_SHUFFLE,
    Behavior.CNC_THETICK: MTDTechnique.CNC_IP_SHUFFLE,
    Behavior.ROOTKIT_BDVL: MTDTechnique.ROOTKIT_SANITIZER,
    Behavior.ROOTKIT_BEURK: MTDTechnique.ROOTKIT_SANITIZER,
    Behavior.RANSOMWARE_POC: MTDTechnique.RANSOMWARE_DIRTRAP
})