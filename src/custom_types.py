from enum import Enum, auto

class RaspberryPi(Enum):
    PI4_2GB_WC = "pi_4_2gb"
    PI3_1GB = "pi_3_1gb"

class Behavior(Enum):
    NORMAL = "normal"
    ROOTKIT_BDVL = "bdvl"
    ROOTKIT_BEURK = "beurk"
    CNC_BACKDOOR_JAKORITAR = "backdoor_jakoritar"
    CNC_THETICK = "the_tick"
    CNC_OPT1 = "data_leak_1"
    CNC_OPT2 = "data_leak_2"
    RANSOMWARE_POC = "ransomware_poc"
   
'''
class Behavior(Enum):
    NORMAL = auto()
    ROOTKIT_BDVL = auto()
    ROOTKIT_BEURK = auto()
    CNC_BACKDOOR_JAKORITAR = auto()
    CNC_THETICK = auto()
    CNC_OPT1 = auto()
    CNC_OPT2 = auto()
    RANSOMWARE_POC = auto()
'''


class MTDTechnique(Enum):
    CNC_IP_SHUFFLE = "cnc_ip_shuffle"
    ROOTKIT_SANITIZER = "rootkit_sanitizer"
    RANSOMWARE_DIRTRAP = "ransomware_directory_trap"
    RANSOMWARE_FILE_EXT_HIDE = "ransomware_file_extension_hide"
      

# define MTD - (target Attack) Mapping
# indices of supervisor_map corresponding to sequence in "actions"
actions = [
    MTDTechnique.CNC_IP_SHUFFLE,
    MTDTechnique.ROOTKIT_SANITIZER,
    MTDTechnique.RANSOMWARE_DIRTRAP,
    MTDTechnique.RANSOMWARE_FILE_EXT_HIDE
]

# noinspection PyTypeChecker
# Mapping between chosen MTD and which attacks can be mitigated by this MTD
mitigated_by = {
    0: [Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK, Behavior.CNC_OPT1, Behavior.CNC_OPT2],
    1: [Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK],
    2: [Behavior.RANSOMWARE_POC],
    3: [Behavior.RANSOMWARE_POC],
}


'''
# noinspection PyTypeChecker
# Mapping between chosen MTD and which attacks can be mitigated by this MTD
supervisor_map = {
    # MTDTechnique.NO_MTD: (Behavior.NORMAL,),
    0: [Behavior.NORMAL],
    1: [Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK, Behavior.CNC_OPT1, Behavior.CNC_OPT2],
    2: [Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK],
    3: [Behavior.RANSOMWARE_POC],
    4: [Behavior.RANSOMWARE_POC],
}
'''

normal_afterstates = [
    (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER),
    (Behavior.ROOTKIT_BEURK, MTDTechnique.ROOTKIT_SANITIZER),
    (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP),
    (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE),
    (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE),
    (Behavior.CNC_THETICK, MTDTechnique.CNC_IP_SHUFFLE),
    (Behavior.CNC_OPT1, MTDTechnique.CNC_IP_SHUFFLE),
    (Behavior.CNC_OPT2, MTDTechnique.CNC_IP_SHUFFLE),
]

'''
from enum import Enum


class RaspberryPi(Enum):
    PI4_2GB_WC = "pi_4_2gb"
    PI3_1GB = "pi_3_1gb"

class Behavior(Enum):
    NORMAL = "normal"
    ROOTKIT_BDVL = "bdvl"
    ROOTKIT_BEURK = "beurk"
    CNC_BACKDOOR_JAKORITAR = "backdoor_jakoritar"
    CNC_THETICK = "the_tick"
    CNC_OPT1 = "data_leak_1"
    CNC_OPT2 = "data_leak_2"
    RANSOMWARE_POC = "ransomware_poc"
    
class MTDTechnique(Enum):
    ROOTKIT_SANITIZER = "rootkit_sanitizer"
    RANSOMWARE_DIRTRAP = "ransomware_directory_trap"
    RANSOMWARE_FILE_EXT_HIDE = "ransomware_file_extension_hide"
    CNC_IP_SHUFFLE = "cnc_ip_shuffle"
    CONTINUE = "continue"
    #NO_MTD = "no_mtd"


# define MTD - (target Attack) Mapping
# indices of supervisor_map corresponding to sequence in "actions"
actions = (MTDTechnique.CNC_IP_SHUFFLE, MTDTechnique.ROOTKIT_SANITIZER,
           MTDTechnique.RANSOMWARE_DIRTRAP, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE)

# noinspection PyTypeChecker
supervisor_map = {
    # MTDTechnique.NO_MTD: (Behavior.NORMAL,),
    0: (Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK, Behavior.CNC_OPT1, Behavior.CNC_OPT2, Behavior.NORMAL),
    1: (Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK, Behavior.NORMAL),
    2: (Behavior.RANSOMWARE_POC, Behavior.NORMAL),
    3: (Behavior.RANSOMWARE_POC, Behavior.NORMAL)
}

normal_afterstates = (
    (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER),
    (Behavior.ROOTKIT_BEURK, MTDTechnique.ROOTKIT_SANITIZER),
    (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP),
    (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE),
    (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE),
    (Behavior.CNC_THETICK, MTDTechnique.CNC_IP_SHUFFLE),
    (Behavior.CNC_OPT1, MTDTechnique.CNC_IP_SHUFFLE),
    (Behavior.CNC_OPT2, MTDTechnique.CNC_IP_SHUFFLE),
)
'''