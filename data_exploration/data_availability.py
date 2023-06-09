import os
from tabulate import tabulate
from data_provider import DataProvider, time_status_columns, all_zero_columns, cols_to_exclude, decision_state, \
    afterstate
from custom_types import Behavior, MTDTechnique


def show_raw_behaviors_data_availability(raw=False, pi=3, decision=False):
    all_data = DataProvider.parse_no_mtd_behavior_data(filter_outliers=not raw,
                                                       filter_suspected_external_events=not raw, pi=pi,
                                                       decision=decision)
    all_data['attack'] = all_data['attack'].apply(lambda x: x.value)

    print(f'Total data points: {len(all_data)}')
    drop_cols = [col for col in list(all_data) if col not in ['attack', 'kmem:kmalloc']]
    grouped = all_data.drop(drop_cols, axis=1).rename(columns={'kmem:kmalloc': 'count'}).groupby(
        ['attack'], as_index=False).count()
    labels = ['Behavior', 'Count']
    rows = []
    for behavior in Behavior:
        row = [behavior.value]
        cnt_row = grouped.loc[(grouped['attack'] == behavior.value)]
        if len(cnt_row) == 0: continue
        row += [cnt_row['count'].iloc[0]]
        rows.append(row)
    print(tabulate(
        rows, headers=labels, tablefmt="latex"))


def show_decision_and_afterstate_data_availability(raw=False):
    all_data = DataProvider.parse_agent_data_files_to_df(filter_outliers=not raw,
                                                         filter_suspected_external_events=not raw)

    print(f'Total agent data points: {len(all_data)}')
    drop_cols = [col for col in list(all_data) if col not in ['attack', 'state', 'block:block_bio_backmerge']]
    grouped = all_data.drop(drop_cols, axis=1).rename(columns={'block:block_bio_backmerge': 'count'}).groupby(
        ['attack', 'state'], as_index=False).count()

    decision_grouped = grouped[grouped['state'] == decision_state]
    after_grouped = grouped[grouped['state'] != decision_state]

    labels = ['Behavior', 'State', 'Count']
    print("decision states data availability")
    rows = []
    for behavior in [b for b in Behavior]:
        line = decision_grouped.loc[(decision_grouped['attack'] == behavior.value), :]
        rows.append(line.values.tolist()[0])
    print(tabulate(
        rows, headers=labels, tablefmt="pretty"))

    print("afterstates data availability")
    rows = []
    for behavior in [b for b in Behavior]:
        lines = after_grouped.loc[(after_grouped['attack'] == behavior.value), :]
        for line in lines.values.tolist():
            rows.append(line)
    print(tabulate(
        rows, headers=labels, tablefmt="latex"))


def print_column_info(raw_behaviors=True, pi=3):
    if raw_behaviors:
        df = DataProvider.parse_no_mtd_behavior_data(filter_suspected_external_events=False,
                                                     filter_constant_columns=False,
                                                     filter_outliers=False, keep_status_columns=True,
                                                     exclude_cols=False,
                                                     pi=pi, decision=False)
    else:
        df = DataProvider.parse_agent_data_files_to_df(filter_suspected_external_events=False,
                                                       filter_constant_columns=False,
                                                       filter_outliers=False, keep_status_columns=True,
                                                       exclude_cols=False)
        df = df.drop(['state'], axis=1)

    df = df.drop(['attack'], axis=1)
    constant_columns = df.loc[:, (df.nunique() <= 1)].columns  # df.columns[df.nunique() <= 1].values
    print("------------------Constant Columns-----------------------")
    print(constant_columns)
    print("---------------------CSV Columns-------------------------")
    labels = ['CSV Column', 'Event Source', 'Event', 'Constant', 'Status', 'Excluded', 'Feature']
    rows = []
    print("Please compare above columns to the marks in table below: "
          "They should be used for the agent data filtering")
    for col in df.columns:
        splitted = col.split(":")
        splind = (col.index(":") if ":" in col else 0) + 3
        splind = min(splind, len(col))
        row = [col if len(splitted) == 1 else (col[:splind] + "..")]
        row += [splitted[0] if (col not in time_status_columns and len(splitted) > 1) else ""]
        row += [(splitted[1] if len(splitted) > 1 else col) if col not in time_status_columns else ""]
        row += ["x" if col in all_zero_columns else ""]
        row += ["x" if col in time_status_columns else ""]
        row += ["x" if col in cols_to_exclude else ""]
        row += ["x" if (col not in time_status_columns and col not in all_zero_columns) else ""]
        rows.append(row)
    print(tabulate(rows[:45], headers=labels, tablefmt='latex'))
    print(tabulate(rows[45:], headers=labels, tablefmt='latex'))


if __name__ == "__main__":
    os.chdir("..")
    print("------------------Raw Data Availability------------------")
    # show_raw_behaviors_data_availability(raw=True, pi=3)
    # show_decision_and_afterstate_data_availability(raw=True)
    print("----------------Filtered Data Availability---------------")
    # show_raw_behaviors_data_availability(raw=False, pi=3)
    # show_decision_and_afterstate_data_availability(raw=False)

    print_column_info(raw_behaviors=True, pi=3)
