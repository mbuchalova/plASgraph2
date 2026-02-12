import pandas as pd

test_list = "/home/buchalova/school/plASgraph2/model/plasgraph2-datasets/eskapee-test.csv"
prefix = "/home/buchalova/school/plASgraph2/model/plasgraph2-datasets/"

rows = []
df = pd.read_csv(test_list, header=None, names=["gfa","truth_csv","sample"])
for _, r in df.iterrows():
    truth = pd.read_csv(prefix + r["truth_csv"])
    truth["sample"] = r["sample"]
    rows.append(truth)

gold = pd.concat(rows, ignore_index=True)

gold[["sample","contig","label","length"]].to_csv("gold_all.csv", index=False)
print("Wrote gold_all.csv with", len(gold), "rows")
