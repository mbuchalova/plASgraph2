#!/usr/bin/env python
# coding: utf-8

import argparse
import inspect
import sys
import pandas as pd
import numpy as np


def row_to_strings(row, decimal_places):
    row2 = list(row)
    for i in range(2, 5):
        row2[i] = f"{float(row2[i]):.0f}"
    for i in range(5, 10):
        row2[i] = f"{float(row2[i]):.{decimal_places}f}"
    return row2


def main(stats_file, summary_type, output_type, decimal_places, additional_columns):
    """Take output from eval.py, selec only rows with summary_type (ALL, MEDIAN or MEAN),
    omit column wth expected number and with stats type, omit decimal places in counts,
    sort by mol_type and metod,
    format everything else with the given number of decimal places and write in a desired format
    (csv, md, txt)

    Additional columns is a comma-separated list of additional columns from the original table
    to include.
    """

    df = pd.read_csv(stats_file)
    df.query("dataset==@summary_type", inplace=True)
    df.sort_values(['mol_type', 'method'], inplace=True)

    columns = ['mol_type', 'method', 'contigs', 'real', 'predicted',
               'auc', 'precision', 'recall', 'f1', 'accuracy']
    if additional_columns is not None:
        columns.extend(additional_columns.split(","))

    for col in columns:
        assert col in df.columns, f"{col} not found in table columns"

    df = df.loc[:, columns]

    if output_type == 'md':
        print("| " + (" | ".join(columns)) + " |")
        print("|" + ((" --- |" * len(columns))))
        for (idx, row) in df.iterrows():
            row2 = row_to_strings(row, decimal_places)
            print("| " + (" | ".join(row2)) + " |")

    if output_type == 'txt':
        print(" " + ("  ".join(columns)))
        for (idx, row) in df.iterrows():
            row2 = row_to_strings(row, decimal_places)
            print(" " + ("  ".join(row2)))

    if output_type == "csv":
        df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=inspect.getdoc(main),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stats_file", help="csv file with output from eval.py")
    parser.add_argument("summary_type", help="ALL, MEDIAN, or MEAN",
                        choices=("ALL", "MEDIAN", "MEAN"))
    parser.add_argument("-o", dest="output_type", help="output format", default='md',
                        choices=['md', 'csv', 'txt'])
    parser.add_argument("-d", dest="decimal_places",
                        type=int, default=3)
    parser.add_argument("-a", dest="additional_columns",
                        help="additional columns (comma-separated)",
                        default=None)
    args = parser.parse_args()
    main(**vars(args))