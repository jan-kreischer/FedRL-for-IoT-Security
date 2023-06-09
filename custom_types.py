from enum import Enum


class Behavior(Enum):
    NORMAL = "normal"
    ROOTKIT_BDVL = "bdvl"
    ROOTKIT_BEURK = "beurk"
    CNC_BACKDOOR_JAKORITAR = "backdoor_jakoritar"
    CNC_THETICK = "the_tick"
    RANSOMWARE_POC = "ransomware_poc"


class MTDTechnique(Enum):
    ROOTKIT_SANITIZER = "rootkit_sanitizer"
    RANSOMWARE_DIRTRAP = "ransomware_directory_trap"
    RANSOMWARE_FILE_EXT_HIDE = "ransomware_file_extension_hide"
    CNC_IP_SHUFFLE = "cnc_ip_shuffle"
    NO_MTD = "no_mtd"
