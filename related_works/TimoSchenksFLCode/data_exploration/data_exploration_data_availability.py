import os

from tabulate import tabulate

from data_handler import DataHandler, time_status_columns, all_zero_columns

if __name__ == "__main__":
    os.chdir("..")
    print("------------------Raw Data Availability------------------")
    DataHandler.show_data_availability(raw=True)
    print("----------------Filtered Data Availability---------------")
    DataHandler.show_data_availability(raw=False)
    print("------------------Constant Columns-----------------------")
    df = DataHandler.parse_all_files_to_df(filter_suspected_external_events=False, filter_constant_columns=False,
                                           filter_outliers=False, keep_status_columns=True)
    df = df.drop(['attack', 'device'], axis=1)
    print(df.columns[df.nunique() <= 1])
    print("---------------------CSV Columns-------------------------")
    labels = ['CSV Column', 'Event Source', 'Event', 'Constant', 'Status', "Feature"]
    rows = []
    for col in df.columns:
        splitted = col.split(":")
        splind = (col.index(":") if ":" in col else 0) + 3
        splind = min(splind, len(col))
        row = [col if len(splitted) == 1 else (col[:splind] + "..")]
        row += [splitted[0] if (col not in time_status_columns and len(splitted) > 1) else ""]
        row += [(splitted[1] if len(splitted) > 1 else col) if col not in time_status_columns else ""]
        row += ["x" if col in all_zero_columns else ""]
        row += ["x" if col in time_status_columns else ""]
        row += ["x" if (col not in time_status_columns and col not in all_zero_columns) else ""]
        rows.append(row)
    print(tabulate(rows[:45], headers=labels, tablefmt='latex'))
    print(tabulate(rows[45:], headers=labels, tablefmt='latex'))
