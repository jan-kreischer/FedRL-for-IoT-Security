import os

from custom_types import Behavior, RaspberryPi, MTDTechnique
from data_plotting import DataPlotter

plot_kde = True
plot_timeline = False

if __name__ == "__main__":
    os.chdir("..")
    if plot_kde:
        # DataPlotter.plot_delay_and_normal_as_kde()
        # DataPlotter.plot_behaviors_as_kde(RaspberryPi.PI4_2GB_WC)
        pass
        # DataPlotter.plot_normals_kde("compare_normals_kde")
        #
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     afterstates=[
        #         (Behavior.NORMAL, MTDTechnique.RANSOMWARE_DIRTRAP, "green"),
        #         (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
        #         (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_DIRTRAP, "lightblue"),
        #         (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_DIRTRAP, "red"),
        #     ],
        #     plot_name="compare_dirtrap_afterstates_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     afterstates=[
        #         (Behavior.NORMAL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "green"),
        #         (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "blue"),
        #         (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
        #         (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "red"),
        #     ],
        #     plot_name="compare_fileextension_afterstates_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     afterstates=[
        #         (Behavior.NORMAL, MTDTechnique.CNC_IP_SHUFFLE, "green"),
        #         (Behavior.RANSOMWARE_POC, MTDTechnique.CNC_IP_SHUFFLE, "blue"),
        #         (Behavior.ROOTKIT_BDVL, MTDTechnique.CNC_IP_SHUFFLE, "lightblue"),
        #         (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE, "red"),
        #     ],
        #     plot_name="compare_changeip_afterstates_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     afterstates=[
        #         (Behavior.NORMAL, MTDTechnique.ROOTKIT_SANITIZER, "green"),
        #         (Behavior.RANSOMWARE_POC, MTDTechnique.ROOTKIT_SANITIZER, "blue"),
        #         (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER, "lightblue"),
        #         (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.ROOTKIT_SANITIZER, "red"),
        #     ],
        #     plot_name="compare_removerk_afterstates_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     [(Behavior.NORMAL, "green")],
        #     [(Behavior.NORMAL, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
        #      (Behavior.NORMAL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
        #      (Behavior.NORMAL, MTDTechnique.CNC_IP_SHUFFLE, "orange"),
        #      (Behavior.NORMAL, MTDTechnique.ROOTKIT_SANITIZER, "red")],
        #     plot_name="dac_normal_decision_to_normal_afterstates_for_mtd_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     [(Behavior.NORMAL, "green")],
        #     [(Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
        #      (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
        #      (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE, "orange"),
        #      (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER, "red")],
        #     plot_name="dac_normal_behavior_to_correct_MTDs_afterstate_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     [(Behavior.RANSOMWARE_POC, "green")],
        #     [(Behavior.RANSOMWARE_POC, MTDTechnique.CNC_IP_SHUFFLE, "blue"),
        #      (Behavior.RANSOMWARE_POC, MTDTechnique.ROOTKIT_SANITIZER, "lightblue")],
        #     plot_name="dac_ransom_behavior_to_incorrect_MTDs_afterstate_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     [(Behavior.CNC_BACKDOOR_JAKORITAR, "green")],
        #     [(Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
        #      (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
        #      (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.ROOTKIT_SANITIZER, "orange")],
        #     plot_name="dac_cnc_behavior_to_incorrect_MTDs_afterstate_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(
        #     [(Behavior.ROOTKIT_BDVL, "green")],
        #     [(Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
        #      (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
        #      (Behavior.ROOTKIT_BDVL, MTDTechnique.CNC_IP_SHUFFLE, "orange")],
        #     plot_name="dac_rootkit_behavior_to_incorrect_MTDs_afterstate_kde")
        #
        # DataPlotter.plot_decision_or_afterstates_as_kde(decision_states=[(Behavior.NORMAL, "green"),
        #                                                                 (Behavior.ROOTKIT_BDVL, "black"),
        #                                                                 (Behavior.CNC_BACKDOOR_JAKORITAR, "blue"),
        #                                                                 (Behavior.RANSOMWARE_POC, "red")],
        #                                                 plot_name="ds_comparison_pi_3_1gb_kde")
        # DataPlotter.plot_devices_as_kde(RaspberryPi.PI4_2GB_WC)

        # UserWarning: Logscale warning can be ignored (some samples have negative values for feature iface0TX)
    if plot_timeline:
        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            afterstates=[
                (Behavior.NORMAL, MTDTechnique.RANSOMWARE_DIRTRAP, "green"),
                (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
                 (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_DIRTRAP, "lightblue"),
                 (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_DIRTRAP, "red"),
                 ],
                plot_name = "compare_dirtrap_afterstates_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            afterstates=[
                (Behavior.NORMAL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "green"),
                (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "blue"),
                (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
                (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "red"),
            ],
            plot_name="compare_fileextension_afterstates_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            afterstates=[
                (Behavior.NORMAL, MTDTechnique.CNC_IP_SHUFFLE, "green"),
                (Behavior.RANSOMWARE_POC, MTDTechnique.CNC_IP_SHUFFLE, "blue"),
                (Behavior.ROOTKIT_BDVL, MTDTechnique.CNC_IP_SHUFFLE, "lightblue"),
                (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE, "red"),
            ],
            plot_name="compare_changeip_afterstates_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            afterstates=[
                (Behavior.NORMAL, MTDTechnique.ROOTKIT_SANITIZER, "green"),
                (Behavior.RANSOMWARE_POC, MTDTechnique.ROOTKIT_SANITIZER, "blue"),
                (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER, "lightblue"),
                (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.ROOTKIT_SANITIZER, "red"),
            ],
            plot_name="compare_removerk_afterstates_timeline")


        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            [(Behavior.NORMAL, "green")],
            [(Behavior.NORMAL, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.NORMAL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.NORMAL, MTDTechnique.CNC_IP_SHUFFLE, "orange"),
             (Behavior.NORMAL, MTDTechnique.ROOTKIT_SANITIZER, "red")],
            plot_name="dac_normal_decision_to_normal_afterstates_for_mtd_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            [(Behavior.NORMAL, "green")],
            [(Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.RANSOMWARE_POC, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.CNC_IP_SHUFFLE, "orange"),
             (Behavior.ROOTKIT_BDVL, MTDTechnique.ROOTKIT_SANITIZER, "red")],
            plot_name="dac_normal_behavior_to_correct_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            [(Behavior.RANSOMWARE_POC, "green")],
            [(Behavior.RANSOMWARE_POC, MTDTechnique.CNC_IP_SHUFFLE, "blue"),
             (Behavior.RANSOMWARE_POC, MTDTechnique.ROOTKIT_SANITIZER, "lightblue")],
            plot_name="dac_ransom_behavior_to_incorrect_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            [(Behavior.CNC_BACKDOOR_JAKORITAR, "green")],
            [(Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.CNC_BACKDOOR_JAKORITAR, MTDTechnique.ROOTKIT_SANITIZER, "orange")],
            plot_name="dac_cnc_behavior_to_incorrect_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(
            [(Behavior.ROOTKIT_BDVL, "green")],
            [(Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_DIRTRAP, "blue"),
             (Behavior.ROOTKIT_BDVL, MTDTechnique.RANSOMWARE_FILE_EXT_HIDE, "lightblue"),
             (Behavior.ROOTKIT_BDVL, MTDTechnique.CNC_IP_SHUFFLE, "orange")],
            plot_name="dac_rootkit_behavior_to_incorrect_MTDs_afterstate_timeline")

        DataPlotter.plot_decision_or_afterstate_behaviors_timeline(decision_states=
                                                                   [(Behavior.NORMAL, "green"),
                                                                    (Behavior.ROOTKIT_BDVL, "black"),
                                                                    (Behavior.CNC_BACKDOOR_JAKORITAR, "blue"),
                                                                    (Behavior.RANSOMWARE_POC, "red")],
                                                                   plot_name="ds_comparison_pi_3_1gb_timeline")

        DataPlotter.plot_behaviors(
            [(RaspberryPi.PI3_1GB, Behavior.NORMAL, "blue"),
             (RaspberryPi.PI3_1GB, Behavior.ROOTKIT_BDVL, "orange"),
             (RaspberryPi.PI3_1GB, Behavior.ROOTKIT_BEURK, "green")],
            plot_name="normal_rootkit_pi_3_1gb_timeline"
        )

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
