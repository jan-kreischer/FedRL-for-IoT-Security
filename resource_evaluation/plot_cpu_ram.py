import pandas as pd


nmon = "ransomware_dirtrap_resourcepi_nmon_220915_1052.csv"

if __name__ == '__main__':
    df = pd.read_csv(nmon)
    print(df)