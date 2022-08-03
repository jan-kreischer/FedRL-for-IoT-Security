import os

from custom_types import Behavior, RaspberryPi
from data_plotter import DataPlotter

plot_kde = True
plot_timeline = True

if __name__ == "__main__":
    os.chdir("..")
    if plot_kde:
        #DataPlotter.plot_delay_and_normal_as_kde()
        #DataPlotter.plot_behaviors_as_kde()
        DataPlotter.plot_devices_as_kde_pub()
    if plot_timeline:
        DataPlotter.plot_behaviors_pub(
            [(RaspberryPi.PI3_1GB, Behavior.NORMAL, "red"),
             (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI4_2GB_BC, Behavior.NORMAL, "orange"),
             (RaspberryPi.PI4_4GB, Behavior.NORMAL, "green")], plot_name="normal_behavior_device_comparison")
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