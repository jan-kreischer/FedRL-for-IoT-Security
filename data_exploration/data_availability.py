import os
from tabulate import tabulate
from data_provider import DataProvider, time_status_columns, all_zero_columns
from custom_types import Behavior




def show_data_availability(raw=False):
    print(os.getcwd())
    all_data = DataProvider.parse_all_files_to_df(filter_outliers=not raw,
                                                  filter_suspected_external_events=not raw)

    print(f'Total data points: {len(all_data)}')
    drop_cols = [col for col in list(all_data) if col not in ['attack', 'block:block_bio_backmerge']]
    grouped = all_data.drop(drop_cols, axis=1).rename(columns={'block:block_bio_backmerge': 'count'}).groupby(
        ['attack'], as_index=False).count()
    labels = ['Behavior', 'Count']
    rows = []
    for behavior in Behavior:
        row = [behavior.value]
        cnt_row = grouped.loc[(grouped['attack'] == behavior.value)]
        row += [cnt_row['count'].iloc[0]]
        rows.append(row)
    print(tabulate(
        rows, headers=labels, tablefmt="pretty"))




if __name__ == "__main__":
    #os.chdir("..")
    print("------------------Raw Data Availability------------------")
    show_data_availability(raw=True)
    print("----------------Filtered Data Availability---------------")
    show_data_availability(raw=False)
    print("------------------Constant Columns-----------------------")
    df = DataProvider.parse_all_files_to_df(filter_suspected_external_events=False, filter_constant_columns=False,
                                            filter_outliers=False, keep_status_columns=True)
    df = df.drop(['attack'], axis=1)
    constant_columns = df.columns[df.nunique() <= 1].values
    print(constant_columns)
    print("---------------------CSV Columns-------------------------")
    labels = ['CSV Column', 'Event Source', 'Event', 'Constant', 'Status', "Feature"]
    rows = []
    print("Check that time_status_columns and all_zero_columns have been specified correctly")
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
    print(tabulate(rows[:45], headers=labels, tablefmt='pretty'))
    print(tabulate(rows[45:], headers=labels, tablefmt='pretty'))