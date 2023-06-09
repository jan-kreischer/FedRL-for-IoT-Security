from src.custom_types import Behavior

unit_sampling_probabilities = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 1/7,
    Behavior.ROOTKIT_BEURK: 1/7,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/7,
    Behavior.CNC_THETICK: 1/7, 
    Behavior.CNC_OPT1: 1/7,
    Behavior.CNC_OPT2: 1/7,
    Behavior.RANSOMWARE_POC: 1/7
}

regular_sampling_probabilities = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 0.12235818,
    Behavior.ROOTKIT_BEURK: 0.17105027,
    Behavior.CNC_BACKDOOR_JAKORITAR: 0.08568661,
    Behavior.CNC_THETICK: 0.17732965,
    Behavior.CNC_OPT1: 0.13046754,
    Behavior.CNC_OPT2: 0.09060246,
    Behavior.RANSOMWARE_POC: 0.22250529,
}

inverted_sampling_probabilities = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 2/7 - 0.12235818,
    Behavior.ROOTKIT_BEURK: 2/7 - 0.17105027,
    Behavior.CNC_BACKDOOR_JAKORITAR: 2/7 - 0.08568661,
    Behavior.CNC_THETICK: 2/7 - 0.17732965, 
    Behavior.CNC_OPT1: 2/7 - 0.13046754,
    Behavior.CNC_OPT2: 2/7 - 0.09060246,
    Behavior.RANSOMWARE_POC: 2/7 - 0.22250529,
}

attack_balanced_sampling_probabilities = {
    Behavior.ROOTKIT_BDVL: 1/7,
    Behavior.ROOTKIT_BEURK: 1/7,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/7,
    Behavior.CNC_THETICK: 1/7, 
    Behavior.CNC_OPT1: 1/7,
    Behavior.CNC_OPT2: 1/7,
    Behavior.RANSOMWARE_POC: 1/7
}

defense_balanced_sampling_probabilities = {
    Behavior.NORMAL: 1/4,
    Behavior.ROOTKIT_BDVL: 1/8,
    Behavior.ROOTKIT_BEURK: 1/8,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/16,
    Behavior.CNC_THETICK: 1/16, 
    Behavior.CNC_OPT1: 1/16,
    Behavior.CNC_OPT2: 1/16,
    Behavior.RANSOMWARE_POC: 1/4
}

weak_client_exclusive_sampling_probabilities_01 = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 1/6,
    Behavior.ROOTKIT_BEURK: 1/6,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/6,
    Behavior.CNC_THETICK: 1/6, 
    Behavior.CNC_OPT1: 1/6,
    Behavior.CNC_OPT2: 0,
    Behavior.RANSOMWARE_POC: 1/6
}

weak_client_exclusive_sampling_probabilities_02 = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 1/6,
    Behavior.ROOTKIT_BEURK: 1/6,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/6,
    Behavior.CNC_THETICK: 1/6, 
    Behavior.CNC_OPT1: 0,
    Behavior.CNC_OPT2: 1/6,
    Behavior.RANSOMWARE_POC: 1/6
}

medium_client_exclusive_sampling_probabilities_01 = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 1/5,
    Behavior.ROOTKIT_BEURK: 1/5,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/5,
    Behavior.CNC_THETICK: 0, 
    Behavior.CNC_OPT1: 1/5,
    Behavior.CNC_OPT2: 0,
    Behavior.RANSOMWARE_POC: 1/5
}

medium_client_exclusive_sampling_probabilities_02 = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 1/5,
    Behavior.ROOTKIT_BEURK: 1/5,
    Behavior.CNC_BACKDOOR_JAKORITAR: 0,
    Behavior.CNC_THETICK: 1/5, 
    Behavior.CNC_OPT1: 0,
    Behavior.CNC_OPT2: 1/5,
    Behavior.RANSOMWARE_POC: 1/5
}

strong_client_exclusive_sampling_probabilities_01 = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 1/4,
    Behavior.ROOTKIT_BEURK: 0,
    Behavior.CNC_BACKDOOR_JAKORITAR: 1/4,
    Behavior.CNC_THETICK: 0, 
    Behavior.CNC_OPT1: 1/4,
    Behavior.CNC_OPT2: 0,
    Behavior.RANSOMWARE_POC: 1/4
}

strong_client_exclusive_sampling_probabilities_02 = {
    #Behavior.NORMAL: 0,
    Behavior.ROOTKIT_BDVL: 0,
    Behavior.ROOTKIT_BEURK: 1/3,
    Behavior.CNC_BACKDOOR_JAKORITAR: 0,
    Behavior.CNC_THETICK: 1/3, 
    Behavior.CNC_OPT1: 0,
    Behavior.CNC_OPT2: 1/3,
    Behavior.RANSOMWARE_POC: 0
}