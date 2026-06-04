import pandas as pd
from scipy.stats import pearsonr

true = pd.read_csv("ground_truth_g.tsv", sep="\t")
est = pd.read_csv("gene_genoinformativity.tsv", sep="\t")

merged = true.merge(est, left_on="gene", right_on="gene")

print(pearsonr(merged["g_true"], merged["g"]))
