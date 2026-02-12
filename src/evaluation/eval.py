#!/usr/bin/env python
# coding: utf-8

import argparse
import inspect
import sys
import pandas as pd
import numpy as np
from sklearn import metrics

from glob import glob

import numpy as np

import helpers


def main(gold_file, prediction_files, names=None, min_length=100, max_length=-1,
         strategy="default", dump_file=None, optimistic=False):
    """Take golden standard answer and several prediction files
    stored in csv files and evaluate per sample and overall prediction accuracy.
    The results are written to standard output.

    Only chromosome and plasmid scores are considered, everything >- 0.5 are counted
    as positive. Text labels are ignored. The only exception is gold answer: if no
    scores are provided, 0/1 scores are filled in according to the text label.

    Evaluation strategies:
    default: Consider two binary classification tasks (chrom, plasmid).
             Contigs unlabeled in gold are omitted.
             Contigs unlabeled in prediction are counted as negative in both tasks.
             Contigs labeled as ambiguous are counted as positive in both tasks.
    ternary: Consider three binary classification tasks (chrom, plasmid, ambiguous).
             Contigs unlabeled in gold are omitted.
             Contigs unlabeled in prediction are counted as negative in all tasks.
             For all other contigs the exact match with correct label is considered.
             AUC is not computed.
    binary:  Consider one binary classification tasks (chrom vs plasmid)
             Contigs unlabeled or ambiguous in gold are omitted.
             For predictions, remaining contigs are put into class with a greater score,
             or to chrom if scores equal (as chrom is assumed to be more common).
    """

    if names is None:
        name_list = prediction_files
    else:
        name_list = names.split(",")
        assert len(name_list) == len(prediction_files)
    if "gold" in name_list:
        raise ValueError("Name 'gold' is reserved for the correct answer, cannot be used for a prediction")

    if strategy == 'binary':
        skip_gold_ambig = True
    else:
        skip_gold_ambig = False

    # read each csv file as pd.DataFrame
    # and method (filename) added
    # store these tables in a list
    gold_df = read_pred_csv(gold_file, "gold")
    gold_df_filtered = filter_gold(gold_df, min_length, max_length, skip_ambig=skip_gold_ambig)
    tables = [gold_df_filtered]
    for (idx, prediction_file) in enumerate(prediction_files):
        tables.append(read_pred_csv(prediction_file, name_list[idx]))

    # concatenate all tables (each has a block of lines)
    big_table = pd.concat(tables, axis=0).reset_index(drop=True)
    samples = big_table["sample"].unique()  # get all samples from the table

    if strategy == "default":
        mol_types = ["plasmid", "chrom"]
    elif strategy == "ternary":
        mol_types = ["plasmid3", "chrom3", "ambig3"]
        transform_scores(big_table, strategy, mol_types)
    elif strategy == "binary":
        mol_types = ["plasmid2", "chrom2"]
        transform_scores(big_table, strategy, mol_types)

    all_results = []  # table of all results
    for method in name_list:
        for mol_type in mol_types:

            # all samples combined
            combined = evaluate(big_table, method, mol_type, "ALL",
                                optimistic=optimistic)
            assert combined is not None
            all_results.append(combined)

            cur_results = []  # list of results to be averaged
            for sample in samples:
                res = evaluate(big_table.query('sample == @sample'),
                               method, mol_type, sample,
                               optimistic=optimistic)
                if res is not None:
                    cur_results.append(res)
                    all_results.append(res)

            # compute median and average
            all_results.append(compute_summary(cur_results, combined, 'MEDIAN'))
            all_results.append(compute_summary(cur_results, combined, 'MEAN'))

    all_results_df = pd.DataFrame.from_records(all_results)

    # blank auc values for ternary as we changed scores to 0/1
    if strategy == "ternary":
        all_results_df['auc'] = np.nan

    all_results_df.to_csv(sys.stdout, index=False)

    if dump_file is not None:
        big_table.to_csv(dump_file, index=False)


def compute_summary(record_list, combined_res, which_summary):
    """Get a list of dictionaries, one for each sample and compute
    summary. Parameter which_summary is MEDIAN or MEAN. Dictionary combined_res
    are results for all samples combined, which are used to determine
    if results are incomplete and should be NaN
    """

    assert len(record_list) > 0
    df = pd.DataFrame.from_records(record_list)

    res = {'dataset': which_summary}

    for column in ['mol_type', 'method']:
        res[column] = record_list[0][column]

    for column in ['contigs', 'real', 'predicted', 'expected']:
        res[column] = df.loc[:, column].sum()

    if pd.isna(combined_res['predicted']):
        # if combined_res does not have value for predicted,
        # replace all values with missing data (some samples were mssing)
        rewrite_columns = ['predicted', 'expected', 'auc', 'precision',
                           'recall', 'f1']
        for column in rewrite_columns:
            res[column] = np.nan
        return res

    for column in ['auc', 'precision', 'recall', 'f1', 'accuracy']:
        if which_summary == 'MEDIAN':
            res[column] = df.loc[:, column].median()
        elif which_summary == 'MEAN':
            res[column] = df.loc[:, column].mean()
        else:
            raise ValueError(f"bad which_summary {which_summary}")

    return res


def evaluate(table, method, mol_type, dataset, optimistic=False):
    value_column = f"{mol_type}_score"

    # get only our method and gold, get only three columns needed
    reduced_table = table.loc[:, ["method", "id", value_column]] \
        .query("method=='gold' or method==@method")

    # no evaluation if no gold answers for this sample
    gold_found = table.query('method=="gold"')
    if gold_found.shape[0] == 0:
        return None

    # transform to a wide table which has methods as columns, indexed by contig id
    wide_table = reduced_table.pivot(index="id", columns="method", values=value_column)
    # drop rows with missing values (unlabeled) in gold
    wide_table.dropna(axis=0, subset=["gold"], inplace=True)

    # get column for gold and start building results
    gold = wide_table.loc[:, "gold"]
    result = {'dataset': dataset, 'mol_type': mol_type, 'method': method}
    result['contigs'] = gold.size
    result['real'] = np.sum(gold)
    for column in ['predicted', 'expected', 'auc', 'precision', 'recall', 'f1', 'accuracy']:
        result[column] = np.nan

    # check for missing values in predictions
    if method not in wide_table.columns:
        print(f"No values found for {method}, {mol_type}, {dataset}", file=sys.stderr)
        return result  # result with missing values

    missing = wide_table.loc[wide_table.isna().any(axis=1), :]
    if missing.shape[0] > 0:
        print(f"Some values ({missing.shape[0]}) missing for {method}, {mol_type}, {dataset}", file=sys.stderr)
        print(f"Several first missing contigs: {' '.join(missing.index[0:5])}", file=sys.stderr)
        return result  # result with missing values

    # get column for prediction
    pred_scores = wide_table.loc[:, method]  # fractional scores
    pred_labels = np.round(pred_scores)  # 0/1 labels

    # build results from prediction
    result['predicted'] = np.sum(pred_labels)
    result['expected'] = np.sum(pred_scores)
    if result['real'] > 0 and result['real'] < result['contigs']:
        result['auc'] = metrics.roc_auc_score(gold, pred_scores)

    if result['predicted'] > 0:
        result['precision'] = metrics.precision_score(gold, pred_labels)
    elif optimistic:
        result['precision'] = 1

    if result['real'] > 0:
        result['recall'] = metrics.recall_score(gold, pred_labels)
    elif optimistic:
        result['recall'] = 1

    # F1 = 2 * (precision * recall) / (precision + recall)
    # define as 0 if both 0; in effect 0 if either zero
    # if one of them undefined, leave undefined
    if np.isfinite(result['precision']) and np.isfinite(result['recall']):
        divide = result['precision'] + result['recall'];
        if divide > 0:
            result['f1'] = 2 * result['precision'] * result['recall'] / divide
        else:
            result['f1'] = 0

    result['accuracy'] = metrics.accuracy_score(gold, pred_labels,
                                                normalize=True)

    return result


def label_to_triple(label):
    if label == "chromosome":
        return [0, 1, 0]
    elif label == "plasmid":
        return [1, 0, 0]
    elif label == "ambiguous":
        return [0, 0, 1]
    elif (label == "unlabelled" or label == "no_label"
          or label == "unlabeled" or label is None):
        return [0, 0, 0]
    else:
        raise AssertionError("bad label {label}")


def transform_scores(df, strategy, mol_types):
    # add new empty columns
    for mol_type in mol_types:
        value_column = f"{mol_type}_score"
        df[value_column] = np.nan

    for idx in df.index:
        pl, chr = df.loc[idx, ['plasmid_score', 'chrom_score']]
        label = helpers.pair_to_label([pl, chr], rounding=True)
        if strategy == 'ternary':
            df.loc[idx, ['plasmid3_score', 'chrom3_score', 'ambig3_score']] = label_to_triple(label)
        elif strategy == 'binary':
            if label == 'ambiguous':
                if chr >= pl:
                    pl = 1 - chr
                else:
                    chr = 1 - pl
            df.loc[idx, ['plasmid2_score', 'chrom2_score']] = [pl, chr]


def filter_gold(df, min_length, max_length, skip_ambig=False):
    """Out of gld answer dataframe, filter out rows with too short or too long
    cntigs and those unlabeled. If desired, ambiguous are skipped as well."""
    assert "length" in df.columns

    # if max_length is -1, use the longest contig as max
    if max_length < 0 and df.shape[0] > 0:
        max_length = df["length"].max() + 1

    df.query("label != 'unlabeled' and length >= @min_length and length <= @max_length", inplace=True)
    if skip_ambig:
        df.query("label != 'ambiguous'", inplace=True)

    return df


def read_pred_csv(csv_file, title):
    df = pd.read_csv(csv_file, header=0, index_col=False, dtype={'sample': 'str', 'contig': 'str'})
    assert "sample" in df.columns and "contig" in df.columns
    df["id"] = df.apply(
        lambda x: helpers.get_node_id(x["sample"], x["contig"]),
        axis=1
    )

    # add scores for gold answer, if needed
    if title == "gold" and "plasmid_score" not in df.columns:
        df["plasmid_score"] = df["label"].apply(lambda x: helpers.label_to_pair(x)[0])
    if title == "gold" and "chrom_score" not in df.columns:
        df["chrom_score"] = df["label"].apply(lambda x: helpers.label_to_pair(x)[1])

    # skip rows with missing scores
    df.dropna(axis=0, subset=["chrom_score", "plasmid_score"], inplace=True)

    # check if some items repeat
    id_counts = df["id"].value_counts()
    if id_counts.shape[0] > 0 and id_counts.iloc[0] > 1:
        raise ValueError(f"repeated contig {id_counts.index[0]} in {title}")

    desired_columns = ["id", "sample", "contig", "method", "label",
                       "plasmid_score", "chrom_score"]
    if title == "gold":
        desired_columns.append("length")
        if 'length' not in df.columns:
            raise KeyError("Gold answer should contain a column named 'length'")

    df["method"] = title
    return df.loc[:, desired_columns]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=inspect.getdoc(main),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gold_file", help="csv file with golden answers")

    parser.add_argument('prediction_files', nargs='+',
                        help='csv files with predictions')
    parser.add_argument("-n", dest="names", help="names of prediction files separated by commas", default=None)
    parser.add_argument("-m", dest="min_length", help="minimum length of contig to include (default 100)",
                        type=int, default=100)
    parser.add_argument("-M", dest="max_length", help="maximum length of contig to include (default -1, no maximum)",
                        type=int, default=-1)
    parser.add_argument("-s", dest="strategy", help="evaluation strategy (default/ternary/binary)",
                        default="default", choices=("default", "ternary", "binary"))
    parser.add_argument("-d", dest="dump_file",
                        help="dump a csv file with all computed scores for binary and ternary",
                        default=None)
    parser.add_argument("-o", dest="optimistic",
                        help="assign precision 1 if nothing predicted and recall 1 if no real positives; otherwise these are undefined",
                        default=False, action='store_true')
    args = parser.parse_args()
    main(**vars(args))