import os

from custom_types import Behavior, RaspberryPi
from data_plotting import DataPlotter

plot_kde = True
plot_timeline = False

if __name__ == "__main__":
    if plot_kde:
        # DataPlotter.plot_delay_and_normal_as_kde()
        # DataPlotter.plot_behaviors_as_kde(RaspberryPi.PI4_2GB_WC)
        DataPlotter.plot_devices_as_kde(RaspberryPi.PI4_2GB_WC)

    # UserWarning: Logscale warning can be ignored (some samples have negative values for feature iface0TX)
    if plot_timeline:
        DataPlotter.plot_behaviors(
            [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_WC, Behavior.ROOTKIT_BDVL, "orange"),
             (RaspberryPi.PI4_2GB_WC, Behavior.ROOTKIT_BEURK, "green"),
             # (RaspberryPi.PI4_2GB_WC, Behavior.CNC_THETICK, "yellow"),
             # (RaspberryPi.PI4_2GB_WC, Behavior.CNC_BACKDOOR_JAKORITAR, "cyan"),
             # (RaspberryPi.PI4_2GB_WC, Behavior.RANSOMWARE_POC, "black")
             ], plot_name="normal_rootkit_pi_4_2gb_timeline")

        DataPlotter.plot_behaviors(
            [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_WC, Behavior.CNC_THETICK, "yellow"),
             (RaspberryPi.PI4_2GB_WC, Behavior.CNC_BACKDOOR_JAKORITAR, "cyan")
             ], plot_name="normal_cnc_pi_4_2gb_timeline")

        DataPlotter.plot_behaviors(
            [(RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_WC, Behavior.RANSOMWARE_POC, "black")
             ], plot_name="normal_ransom_pi_4_2gb_timeline")
