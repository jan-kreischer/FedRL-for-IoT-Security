from enum import Enum, auto

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
    CNC_IP_SHUFFLE = "cnc_ip_shuffle"
    ROOTKIT_SANITIZER = "rootkit_sanitizer"
    RANSOMWARE_DIRTRAP = "ransomware_directory_trap"
    RANSOMWARE_FILE_EXT_HIDE = "ransomware_file_extension_hide"
      
actions = [
    MTDTechnique.CNC_IP_SHUFFLE,
    MTDTechnique.ROOTKIT_SANITIZER,
    MTDTechnique.RANSOMWARE_DIRTRAP,
    MTDTechnique.RANSOMWARE_FILE_EXT_HIDE
]

mitigated_by = {
    0: [Behavior.CNC_BACKDOOR_JAKORITAR, Behavior.CNC_THETICK, Behavior.CNC_OPT1, Behavior.CNC_OPT2],
    1: [Behavior.ROOTKIT_BDVL, Behavior.ROOTKIT_BEURK],
    2: [Behavior.RANSOMWARE_POC],
    3: [Behavior.RANSOMWARE_POC],
}

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