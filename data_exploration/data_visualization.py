import os

from custom_types import Behavior, RaspberryPi, MTDTechnique
from data_plotting import DataPlotter

plot_kde = False
plot_timeline = True

if __name__ == "__main__":
    os.chdir("..")
    if plot_kde:
        # DataPlotter.plot_delay_and_normal_as_kde()
        # DataPlotter.plot_behaviors_as_kde(RaspberryPi.PI4_2GB_WC)
        DataPlotter.plot_devices_as_kde(RaspberryPi.PI4_2GB_WC)

    # UserWarning: Logscale warning can be ignored (some samples have negative values for feature iface0TX)
    if plot_timeline:
        DataPlotter.plot_decision_afterstate_comparison(
            [(Behavior.NORMAL, "green")],
            [(Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE, "orange"),
             (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER, "red")],
            plot_name="dac_normal_behavior_to_correct_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_afterstate_comparison(
            [(Behavior.RANSOMWARE_POC, "green")],
            [(Behavior.RANSOMWARE_POC, MTDTechnique.CNC_IP_SHUFFLE, "blue"),
             (Behavior.RANSOMWARE_POC, MTDTechnique.ROOTKIT_SANITIZER, "lightblue")],
            plot_name="dac_ransom_behavior_to_incorrect_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_afterstate_comparison(
            [(Behavior.CNC_BACKDOOR_JAKORITAR, "green")],
            [(Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.ROOTKIT_SANITIZER, "orange")],
            plot_name="dac_cnc_behavior_to_incorrect_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_afterstate_comparison(
            [(Behavior.ROOTKIT_BDVL, "green")],
            [(Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.ROOTKIT_BDVL, MTDTechnique.CNC_IP_SHUFFLE, "orange")],
            plot_name="dac_rootkit_behavior_to_incorrect_MTDs_afterstate_timeline")

        # DataPlotter.plot_behaviors(
        #     [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.ROOTKIT_BDVL, "orange"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.ROOTKIT_BEURK, "green"),
        #      # (RaspberryPi.PI4_2GB_WC, Behavior.CNC_THETICK, "yellow"),
        #      # (RaspberryPi.PI4_2GB_WC, Behavior.CNC_BACKDOOR_JAKORITAR, "cyan"),
        #      # (RaspberryPi.PI4_2GB_WC, Behavior.RANSOMWARE_POC, "black")
        #      ], plot_name="normal_rootkit_pi_4_2gb_timeline")
        #
        # DataPlotter.plot_behaviors(
        #     [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.CNC_THETICK, "yellow"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.CNC_BACKDOOR_JAKORITAR, "cyan")
        #      ], plot_name="normal_cnc_pi_4_2gb_timeline")
        #
        # DataPlotter.plot_behaviors(
        #     [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.RANSOMWARE_POC, "black")
        #      ], plot_name="normal_ransom_pi_4_2gb_timeline")
