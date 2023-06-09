import os

from custom_types import Behavior, RaspberryPi
from data_plotting import DataPlotter

plot_kde = False
plot_timeline = True

if __name__ == "__main__":
    if plot_kde:pass
        #DataPlotter.plot_delay_and_normal_as_kde()
        #DataPlotter.plot_behaviors_as_kde()
        #DataPlotter.plot_devices_as_kde_pub()
    if plot_timeline:
        DataPlotter.plot_behaviors(#_pub(
            [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_WC, Behavior.ROOTKIT_BDVL, "orange"),
             (RaspberryPi.PI4_2GB_WC, Behavior.ROOTKIT_BEURK, "green"),
             #(RaspberryPi.PI4_2GB_WC, Behavior.CNC_THETICK, "yellow"),
             #(RaspberryPi.PI4_2GB_WC, Behavior.CNC_BACKDOOR_JAKORITAR, "cyan"),
             #(RaspberryPi.PI4_2GB_WC, Behavior.RANSOMWARE_POC, "black")
             ], plot_name="normal_rootkit_pi_4_2gb_histogram")

        DataPlotter.plot_behaviors(  # _pub(
            [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_WC, Behavior.CNC_THETICK, "yellow"),
             (RaspberryPi.PI4_2GB_WC, Behavior.CNC_BACKDOOR_JAKORITAR, "cyan"),
             ], plot_name="normal_cnc_pi_4_2gb_histogram")

        DataPlotter.plot_behaviors(  # _pub(
            [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_WC, Behavior.RANSOMWARE_POC, "black"),
             ], plot_name="normal_ransom_pi_4_2gb_histogram")

        #
        # DataPlotter.plot_behaviors(
        #     [(RaspberryPi.PI4_2GB_WC, Behavior.HOP, "darkred"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.NOISE, "red"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.SPOOF, "yellow"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.DELAY, "goldenrod"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.DISORDER, "cyan"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.FREEZE, "black"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.REPEAT, "blue"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.MIMIC, "fuchsia"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "lightgreen"),
        #      (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL_V2, "darkgreen")], plot_name="all_pi4_2gb_hist")